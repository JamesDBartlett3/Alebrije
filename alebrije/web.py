"""Local web UI: a small ofrenda for your archive, served on your LAN.

stdlib http.server + one embedded single-page app, themed for Día de los
Muertos — marigold and cempasúchil orange, papel picado banners, warm
candlelight. Any phone or laptop on the same network can open it.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import socket
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import db, people, places, search, vectors

_CACHE: dict = {}


def _cached(key: str, ttl: float, fn):
    import time
    hit = _CACHE.get(key)
    now = time.time()
    if hit and now - hit[0] < ttl:
        return hit[1]
    val = fn()
    _CACHE[key] = (now, val)
    return val


def _media_payload(conn, row) -> dict:
    d = db.media_row_to_dict(row)
    d["people"] = [r["name"] for r in conn.execute(
        "SELECT p.name FROM person_tags t JOIN people p ON p.id=t.person_id "
        "WHERE t.media_id=? ORDER BY p.name", (row["id"],))]
    return d


def api_media(conn, cfg, query: dict) -> dict:
    res = search.search(
        conn, cfg, query.get("q", [""])[0],
        sort=query.get("sort", ["relevance"])[0],
        media_type=query.get("type", [None])[0],
        person=query.get("person", [None])[0],
        limit=int(query.get("limit", ["60"])[0]))
    return res


def api_people(conn) -> dict:
    return {"people": people.list_people(conn)}


def api_stats(conn) -> dict:
    total = conn.execute("SELECT COUNT(*) c FROM media").fetchone()["c"]
    tagged = conn.execute("SELECT COUNT(*) c FROM media WHERE tagged=1").fetchone()["c"]
    with_gps = conn.execute(
        "SELECT COUNT(*) c FROM media WHERE gps_lat IS NOT NULL").fetchone()["c"]
    with_place = conn.execute(
        "SELECT COUNT(*) c FROM media WHERE place IS NOT NULL AND place != ''").fetchone()["c"]
    return {"total": total, "tagged": tagged, "with_gps": with_gps,
            "with_place": with_place, "embedded": vectors.index_count(conn),
            "people": conn.execute("SELECT COUNT(*) c FROM people").fetchone()["c"]}


def api_add_person(conn, cfg, body: dict) -> dict:
    name = str(body.get("name", "")).strip()
    desc = str(body.get("description", "")).strip()
    if not name or not desc:
        return {"ok": False, "error": "need name and description"}
    pid = people.add_person(conn, name, desc, body.get("ref_media"))
    named = people.name_people_in_media(conn, cfg, log=lambda *_: None)
    return {"ok": True, "id": pid, "matched": named.get("named", 0)}


def _serve_file(handler, path: str, archive_root: str) -> bool:
    """Serve a media file from inside the archive root only."""
    full = os.path.realpath(os.path.join(archive_root, path))
    root = os.path.realpath(archive_root)
    if not full.startswith(root + os.sep) and full != root:
        return False
    if not os.path.isfile(full):
        return False
    ctype = mimetypes.guess_type(full)[0] or "application/octet-stream"
    with open(full, "rb") as fh:
        data = fh.read()
    handler.send_response(200)
    handler.send_header("Content-Type", ctype)
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)
    return True


def _lan_urls(port: int) -> list[str]:
    urls = []
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ip = s.getsockname()[0]
        s.close()
        urls.append(f"http://{ip}:{port}")
    except OSError:
        pass
    return urls


class BattenHandler(BaseHTTPRequestHandler):
    server_version = "alebrije/0.1"

    cfg: dict = {}
    conn = None

    def log_message(self, fmt, *args):  # quieter default
        pass

    def _json(self, obj, code: int = 200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        parts = parsed.path.strip("/").split("/")
        q = urllib.parse.parse_qs(parsed.query)
        conn = self.conn
        cfg = self.cfg
        if parsed.path in ("/", "/index.html"):
            body = INDEX_HTML.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif parts[:1] == ["api"]:
            route = parts[1] if len(parts) > 1 else ""
            if route == "media":
                self._json(api_media(conn, cfg, q))
            elif route == "people":
                self._json(api_people(conn))
            elif route == "stats":
                self._json(_cached("stats", 5, lambda: api_stats(conn)))
            elif route == "file":
                if not _serve_file(self, q.get("path", [""])[0], cfg["archive_dir"]):
                    self._json({"error": "not found"}, 404)
            elif route == "places":
                n = places.backfill(conn, cfg, limit=10, log=lambda *_: None)
                self._json({"ok": True, "named": n})
            else:
                self._json({"error": "unknown endpoint"}, 404)
        else:
            self.send_error(404)

    def do_POST(self):
        parts = urllib.parse.urlparse(self.path).path.strip("/").split("/")
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json({"error": "bad request body"}, 400)
            return
        if parts[:2] == ["api", "person"]:
            self._json(api_add_person(self.conn, self.cfg, body))
        elif parts[:2] == ["api", "search"]:
            res = search.search(self.conn, self.cfg, body.get("q", ""),
                                sort=body.get("sort", "relevance"),
                                media_type=body.get("type"),
                                person=body.get("person"),
                                limit=int(body.get("limit", 60)))
            self._json(res)
        else:
            self._json({"error": "unknown endpoint"}, 404)


def serve(cfg: dict, host: str | None = None, port: int | None = None):
    host = host or cfg.get("web_host", "0.0.0.0")
    port = int(port or cfg.get("web_port", 8799))
    conn = db.connect(require_db(cfg))
    BattenHandler.cfg = cfg
    BattenHandler.conn = conn
    httpd = ThreadingHTTPServer((host, port), BattenHandler)
    print(f"🌼 Alebrije is live — the ofrenda is open.")
    print(f"   this machine:  http://localhost:{port}")
    for u in _lan_urls(port):
        print(f"   on your LAN:   {u}  (any phone or laptop on the same network)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🕯️  closing the ofrenda — buenas noches")
    finally:
        conn.close()


def require_db(cfg) -> str:
    from .config import db_path
    return db_path(cfg)


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>🌼 Alebrije — a spirit guide for your memories</title>
<style>
:root{
  --marigold:#f39c12; --marigold-deep:#d35400; --cempa:#ff9f1c;
  --purple:#6c3483; --purple-deep:#4a235a; --night:#170d24; --card:#241538;
  --cream:#fdf3e0; --paper:#fff8ec; --glow:rgba(243,156,18,.35);
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--night);color:var(--cream);
  font:16px/1.5 -apple-system,'Segoe UI',Roboto,sans-serif;min-height:100vh}
.papel{height:34px;display:flex;overflow:hidden}
.papel i{flex:1;clip-path:polygon(0 0,100% 0,100% 72%,50% 100%,0 72%);
  background:var(--purple);margin-right:6px}
.papel i:nth-child(2n){background:var(--marigold)}
.papel i:nth-child(3n){background:#c0392b}
.papel i:nth-child(5n){background:#16a085}
header{text-align:center;padding:18px 12px 6px}
h1{font-size:1.7rem;color:var(--cempa);text-shadow:0 0 18px var(--glow)}
h1 span{color:var(--paper)}
.tagline{color:#c9a7e8;font-size:.95rem;margin-top:2px}
.stats{display:flex;gap:14px;justify-content:center;margin:10px 0;flex-wrap:wrap}
.stats b{color:var(--cempa)}
.stats span{font-size:.85rem;color:#b89ad6}
.searchbar{display:flex;gap:8px;max-width:860px;margin:14px auto;padding:0 12px}
.searchbar input[type=search]{flex:1;padding:12px 16px;border-radius:999px;border:1px solid var(--purple);
  background:var(--card);color:var(--cream);font-size:1rem;outline:none}
.searchbar input[type=search]:focus{border-color:var(--cempa);box-shadow:0 0 14px var(--glow)}
button,.chip{cursor:pointer;border:none;border-radius:999px;padding:9px 16px;font-size:.9rem;
  background:var(--marigold);color:#241505;font-weight:600}
button:hover{background:var(--cempa)}
.chip{background:var(--card);color:var(--cream);border:1px solid var(--purple)}
.chip.on{background:var(--marigold);color:#241505;border-color:var(--marigold)}
.filters{display:flex;gap:8px;justify-content:center;flex-wrap:wrap;padding:0 12px}
.grid{columns:4 240px;column-gap:12px;max-width:1200px;margin:18px auto;padding:0 12px}
.card{break-inside:avoid;margin-bottom:12px;background:var(--card);border-radius:14px;overflow:hidden;
  border:1px solid #3a2455;cursor:pointer;transition:transform .15s, box-shadow .15s}
.card:hover{transform:translateY(-3px);box-shadow:0 8px 24px var(--glow)}
.card img,.card .ph{width:100%;display:block;object-fit:cover;background:var(--purple-deep)}
.card .ph{aspect-ratio:4/3;display:flex;align-items:center;justify-content:center;font-size:2.4rem}
.card .meta{padding:8px 10px}
.card .cap{font-size:.85rem;color:var(--paper)}
.card .sub{font-size:.72rem;color:#a98cc7;margin-top:3px;display:flex;gap:6px;flex-wrap:wrap}
.badge{background:var(--purple-deep);border-radius:6px;padding:1px 6px}
.badge.person{background:var(--marigold-deep);color:#fff}
dialog{background:var(--card);color:var(--cream);border:1px solid var(--purple);border-radius:16px;
  max-width:820px;width:92vw;padding:0}
dialog::backdrop{background:rgba(23,13,36,.85)}
#detail img{width:100%;max-height:52vh;object-fit:contain;background:#120a1e}
#detail .body{padding:14px 18px 20px}
#detail h3{color:var(--cempa);margin-bottom:6px}
#detail .row{margin:6px 0;font-size:.92rem}
#detail .k{color:#a98cc7;margin-right:6px}
footer{text-align:center;padding:26px;color:#8a6fae;font-size:.8rem}
.empty{text-align:center;color:#a98cc7;padding:60px 20px}
a{color:var(--cempa)}
@media(max-width:640px){h1{font-size:1.3rem}.grid{columns:2 150px}}
</style>
</head>
<body>
<div class="papel" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div>
<header>
  <h1>🌼 <span>Alebrije</span></h1>
  <div class="tagline">a spirit guide for your memories — everything local, nothing leaves this machine</div>
  <div class="stats" id="stats"></div>
</header>
<div class="searchbar">
  <input id="q" type="search" placeholder="search your memories… “abuela laughing at the beach”" autocomplete="off">
  <button onclick="runSearch()">Search</button>
</div>
<div class="filters">
  <span class="chip on" data-sort="relevance" onclick="pickSort(this)">relevance</span>
  <span class="chip" data-sort="date" onclick="pickSort(this)">newest</span>
  <span class="chip" data-sort="type" onclick="pickSort(this)">type</span>
  <span class="chip" data-sort="name" onclick="pickSort(this)">name</span>
  <span class="chip" data-type="" onclick="pickType(this)">all</span>
  <span class="chip" data-type="image" onclick="pickType(this)">photos</span>
  <span class="chip" data-type="video" onclick="pickType(this)">videos</span>
  <span class="chip" data-type="audio" onclick="pickType(this)">audio</span>
  <span class="chip" onclick="showPeople()">👤 people</span>
</div>
<div class="grid" id="grid"></div>
<div class="empty" id="empty" hidden></div>
<footer>🕯️ built with open-weight AI (Gemma 3 · nomic-embed · llama.cpp) — Día de los Muertos, every day</footer>

<dialog id="detail"><div id="detailInner"></div></dialog>
<dialog id="peopleDlg">
  <div style="padding:18px">
    <h3 style="color:var(--cempa)">👤 People book</h3>
    <div id="peopleList" style="margin:12px 0"></div>
    <div style="display:flex;gap:8px;flex-wrap:wrap">
      <input id="npName" placeholder="Name" style="flex:1;min-width:130px;padding:10px;border-radius:8px;border:1px solid var(--purple);background:var(--night);color:var(--cream)">
      <input id="npDesc" placeholder="Description, e.g. “older woman, white braid, jade earrings”" style="flex:2;min-width:220px;padding:10px;border-radius:8px;border:1px solid var(--purple);background:var(--night);color:var(--cream)">
      <button onclick="addPerson()">Add</button>
      <span class="chip" onclick="document.getElementById('peopleDlg').close()">Close</span>
    </div>
  </div>
</dialog>
<script>
let sort="relevance", type="", lastQ="";
const grid=document.getElementById("grid"), empty=document.getElementById("empty");
function esc(s){return (s||"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]))}
async function api(path, opts){const r=await fetch(path,opts);return r.json()}
async function loadStats(){
  const s=await api("/api/stats");
  document.getElementById("stats").innerHTML =
    `<span><b>${s.total}</b> memories</span><span><b>${s.tagged}</b> tagged</span>`+
    `<span><b>${s.people}</b> people known</span><span><b>${s.with_place}</b> places named</span>`;
}
async function runSearch(q){
  if(q!==undefined) lastQ=q;
  const p=new URLSearchParams({q:lastQ, sort, limit:120}); if(type)p.set("type",type);
  const res=await api("/api/media?"+p);
  grid.innerHTML=""; 
  const items=res.results||[];
  empty.hidden=items.length>0;
  empty.textContent = items.length? "" :
    "Nothing here yet. Run `alebrije scan` and `alebrije tag` to introduce me to your archive. 🌼";
  for(const m of items){
    const el=document.createElement("div"); el.className="card";
    const icon={image:"🖼️",video:"🎬",audio:"🎵"}[m.media_type]||"📁";
    const sub=[m.media_type, (m.created_at||"").slice(0,10)]
      .concat((m.people||[]).map(n=>`<span class="badge person">${esc(n)}</span>`))
      .concat((m.tags||[]).slice(0,3).map(t=>`<span class="badge">${esc(t)}</span>`));
    el.innerHTML = `${m.media_type==="image"
        ? `<img loading="lazy" src="/api/file?path=${encodeURIComponent(m.path)}" alt="">`
        : `<div class="ph">${icon}</div>`}
      <div class="meta"><div class="cap">${esc(m.caption||m.filename)}</div>
      <div class="sub">${sub.map(s=>`<span>${s}</span>`).join("")}</div></div>`;
    el.onclick=()=>showDetail(m);
    if(new URLSearchParams(location.search).get("open")===String(m.id)) setTimeout(()=>showDetail(m),300);
    grid.appendChild(el);
  }
}
async function showDetail(m){
  const dlg=document.getElementById("detail"), inner=document.getElementById("detailInner");
  const icon={image:"🖼️",video:"🎬",audio:"🎵"}[m.media_type]||"📁";
  const fileUrl = "/api/file?path="+encodeURIComponent(m.path);
  inner.innerHTML = `
    <div id="detail" class="detailwrap">
      ${m.media_type==="image"
        ? `<img src="${fileUrl}" alt="">`
        : m.media_type==="video"
          ? `<video src="${fileUrl}" controls style="width:100%;max-height:52vh;background:#120a1e"></video>`
          : `<div style="padding:40px;text-align:center;font-size:3rem">${icon}</div><audio src="${fileUrl}" controls style="width:100%"></audio>`}
      <div class="body">
        <h3>${esc(m.caption||m.filename)}</h3>
        <div class="row"><span class="k">file</span>${esc(m.filename)}</div>
        ${m.place?`<div class="row"><span class="k">📍 place</span>${esc(m.place)}</div>`:""}
        ${m.created_at?`<div class="row"><span class="k">🗓️ taken</span>${esc(m.created_at.slice(0,10))}</div>`:""}
        ${m.mood?`<div class="row"><span class="k">🌼 mood</span>${esc(m.mood)}</div>`:""}
        ${(m.people||[]).length?`<div class="row"><span class="k">👤 people</span>${m.people.map(esc).join(", ")}</div>`:""}
        ${(m.tags||[]).length?`<div class="row"><span class="k">🏷️ tags</span>${m.tags.map(esc).join(" · ")}</div>`:""}
        <div class="row" style="margin-top:12px"><span class="chip" onclick="document.getElementById('detail').close()">Close</span></div>
      </div></div>`;
  dlg.showModal();
}
async function showPeople(){
  const dlg=document.getElementById("peopleDlg");
  const res=await api("/api/people");
  document.getElementById("peopleList").innerHTML =
    (res.people||[]).map(p=>`<div>👤 <b style="color:var(--cempa)">${esc(p.name)}</b> — ${esc(p.description)} <span style="color:#8a6fae">(${p.tagged_media} media)</span></div>`).join("")
    || "<i style='color:#a98cc7'>Nobody introduced yet. Add someone from a sample photo's description.</i>";
  dlg.showModal();
}
async function addPerson(){
  const name=document.getElementById("npName").value.trim(),
        description=document.getElementById("npDesc").value.trim();
  if(!name||!description)return;
  await api("/api/person",{method:"POST",headers:{"Content-Type":"application/json"},
    body:JSON.stringify({name,description})});
  document.getElementById("npName").value="";document.getElementById("npDesc").value="";
  await showPeople(); await loadStats(); await runSearch(lastQ);
}
function pickSort(el){sort=el.dataset.sort;
  document.querySelectorAll("[data-sort]").forEach(c=>c.classList.toggle("on",c===el));runSearch(lastQ)}
function pickType(el){type=el.dataset.type;
  document.querySelectorAll("[data-type]").forEach(c=>c.classList.toggle("on",c===el));runSearch(lastQ)}
document.getElementById("q").addEventListener("keydown",e=>{if(e.key==="Enter")runSearch(e.target.value)});
const v0=new URLSearchParams(location.search).get("q"); if(v0){document.getElementById("q").value=v0}
loadStats(); runSearch(v0||"");
</script>
</body>
</html>"""
