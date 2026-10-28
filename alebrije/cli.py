"""The alebrije command line: init, setup, up, scan, tag, person, search, serve."""

from __future__ import annotations

import argparse
import json
import os
import sys

if __package__ in (None, ""):  # run as a plain file: `python alebrije/cli.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from alebrije import (bootstrap, config, db, llm, people, places, scanner,
                          search, selftest, servers, tagger, update, vectors)
else:
    from . import (bootstrap, config, db, llm, people, places, scanner, search,
                   selftest, servers, tagger, update, vectors)
from alebrije.errors import LlmError


def _db(cfg):
    conn = db.connect(config.db_path(cfg))
    return conn


def _print_results(res: dict):
    for r in res["results"]:
        line = (f"  {(r['score'] if r['score'] is not None else 0):6.3f}  "
                f"{(r['created_at'] or ''):19.19}  {r['media_type'][:2]:2}  {r['filename']}")
        if r["people"]:
            line += f"  [{', '.join(r['people'])}]"
        caption = (r.get("caption") or "")[:70]
        if caption:
            line += f"  {caption}"
        print(line)


def cmd_init(cfg, args) -> int:
    if args.archive:
        archive = os.path.abspath(os.path.expanduser(args.archive))
        if not os.path.isdir(archive):
            print(f"❌ no such directory: {archive}")
            return 1
        cfg = config.update(cfg, archive_dir=archive)
    if not cfg["archive_dir"]:
        print("Usage: alebrije init <archive-directory>")
        return 1
    config.update(cfg, home=args.home or cfg["home"],
                  vlm_url=args.vlm_url, embed_url=args.embed_url,
                  web_port=args.port)
    config.save(cfg)
    _db(cfg)  # creates the database
    print(f"🌼 Alebrije is watching {cfg['archive_dir']}")
    print(f"   state: {config.home_dir(cfg)}")
    if not bootstrap.llama_server_path(cfg):
        print("   next: `alebrije setup` — downloads llama.cpp + the open-weight "
              "models automatically (GPU auto-detected).")
    print("   next: `alebrije up && alebrije scan`")
    return 0


def cmd_setup(cfg, args) -> int:
    try:
        py = bootstrap.check_python()
        print(f"🐍 Python {py} — OK")
    except RuntimeError as exc:
        print(f"❌ {exc}")
        return 1
    info = bootstrap.setup_all(cfg, log=print)
    print(f"🌺 everything installed. server: {info['server']}")
    print("   next: `alebrije init <archive>` (if you haven't) then `alebrije up`")
    return 0


def cmd_up(cfg, args) -> int:
    try:
        servers.start(cfg, log=print, wait_secs=args.wait)
        return 0
    except (RuntimeError, TimeoutError) as exc:
        print(f"❌ {exc}")
        return 1


def cmd_stop(cfg, args) -> int:
    servers.stop(cfg, log=print)
    return 0


def cmd_scan(cfg, args) -> int:
    if not cfg["archive_dir"]:
        print("❌ run `alebrije init <archive-directory>` first")
        return 1
    conn = _db(cfg)
    res = scanner.scan_archive(cfg["archive_dir"], conn, _progress=lambda fn, a, u, uch: None)
    print(f"🗂️  scanned {cfg['archive_dir']}: +{res['added']} new, ~{res['updated']} changed, "
          f"{res['unchanged']} unchanged, −{res['removed']} gone")
    n = places.backfill(conn, cfg, limit=25, log=print)
    if n:
        print(f"📍 named {n} places from GPS coordinates")
    conn.close()
    return 0


def cmd_tag(cfg, args) -> int:
    conn = _db(cfg)
    res = tagger.tag_all(conn, cfg, limit=args.limit, log=print)
    print(f"🌼 tagged {res['tagged']}, failed {res['failed']}, "
          f"{res['remaining']} remaining")
    if res["tagged"]:
        idx = vectors.index_media(conn, cfg, log=lambda *_: None)
        print(f"🧠 embedded {idx} memories into the search index")
        named = people.name_people_in_media(conn, cfg, log=lambda *_: None)
        if named.get("named"):
            print(f"👥 matched {named['named']} people appearances")
    conn.close()
    return 0


def cmd_person(cfg, args) -> int:
    conn = _db(cfg)
    if args.which == "add":
        if not args.name:
            print("Usage: alebrije person add <Name> --photo <sample.jpg> [--note \"...\"]")
            return 1
        desc = args.note or ""
        if args.photo and not desc:
            photo = os.path.abspath(os.path.expanduser(args.photo))
            if not os.path.isfile(photo):
                print(f"❌ no such photo: {photo}")
                return 1
            try:
                raw = llm.extract_json(llm.describe_image(cfg, photo, tagger.TAG_PROMPT))
                people_seen = raw.get("people") or []
            except LlmError as exc:
                print(f"❌ couldn't read the sample photo ({exc.kind})")
                return 1
            if not people_seen:
                print("❌ no person found in that photo — try one where their face "
                      "is clearly visible")
                return 1
            if len(people_seen) > 1:
                print("There is more than one person in that photo. Which one is "
                      f"{args.name}? Give a short note to pin it down:")
                for i, p in enumerate(people_seen):
                    print(f"  {i}: {p.get('description')}")
                pick = input("number: ").strip()
                try:
                    desc = str(people_seen[int(pick)].get("description", ""))
                except (ValueError, IndexError):
                    print("cancelled")
                    return 1
            else:
                desc = str(people_seen[0].get("description", ""))
        pid = people.add_person(conn, args.name, desc, args.photo)
        print(f"👤 added {args.name} to the people book (#{pid}): {desc}")
        # immediately apply the new name across the archive
        named = people.name_people_in_media(conn, cfg, log=lambda *_: None)
        if named.get("named"):
            print(f"👥 matched {named['named']} appearances across the archive")
    elif args.which == "list":
        book = people.list_people(conn)
        if not book:
            print("The people book is empty. `alebrije person add <Name> --photo <sample.jpg>`")
        for p in book:
            print(f"  👤 {p['name']} — {p['description']} ({p['tagged_media']} media)")
    elif args.which == "rename":
        if not (args.name and args.new):
            print("Usage: alebrije person rename <OldName> <NewName>")
            return 1
        conn.execute("UPDATE people SET name=? WHERE name=?", (args.new, args.name))
        conn.commit()
        print(f"✍️  {args.name} is now {args.new}")
    conn.close()
    return 0


def cmd_name(cfg, args) -> int:
    conn = _db(cfg)
    res = people.name_people_in_media(conn, cfg, limit=args.limit, log=print)
    print(f"👥 {res}")
    conn.close()
    return 0


def cmd_places(cfg, args) -> int:
    conn = _db(cfg)
    n = places.backfill(conn, cfg, limit=args.limit, log=print)
    print(f"📍 named {n} places")
    conn.close()
    return 0


def cmd_search(cfg, args) -> int:
    conn = _db(cfg)
    res = search.search(conn, cfg, " ".join(args.query), sort=args.sort,
                        media_type=args.type, person=args.person, limit=args.limit)
    print(f"🔎 {res['mode']} search, sorted by {res['sort']} — {len(res['results'])} results")
    _print_results(res)
    conn.close()
    return 0


def cmd_serve(cfg, args) -> int:
    from . import web
    cfg = config.update(cfg, web_host=args.host, web_port=args.port)
    web.serve(cfg)
    return 0


def cmd_doctor(cfg, args) -> int:
    ok = True
    print("🩺 Alebrije doctor")
    py = ".".join(map(str, sys.version_info[:2]))
    print(f"  {'✅' if sys.version_info >= (3, 10) else '❌'} Python {py} (needs 3.10+)")
    ok &= sys.version_info >= (3, 10)
    server = bootstrap.llama_server_path(cfg)
    print(f"  {'✅' if server else '❌'} llama.cpp: {server or 'not installed — `alebrije setup`'}")
    ok &= bool(server)
    mdir = bootstrap.model_dir(cfg)
    for m in bootstrap.MODELS:
        present = os.path.isfile(os.path.join(mdir, m["file"]))
        print(f"  {'✅' if present else '⏳'} model {m['name']}")
    from . import accel
    info = accel.detect()
    print(f"  ⚡ accelerator: {info['backend']} ({info['note']})")
    up = servers.is_up(cfg)
    for label, url, good in (("vision", cfg["vlm_url"], up["vlm"]),
                             ("embeddings", cfg["embed_url"], up["embed"])):
        print(f"  {'✅' if good else '⏳'} {label} server {url}")
    try:
        vec = vectors.embed_one(cfg, "hello")
        print(f"  ✅ embeddings respond ({len(vec)} dims)")
    except LlmError as exc:
        print(f"  ⏳ embeddings: {exc.kind}")
    try:
        llm.chat(cfg, [{"role": "user", "content": "Reply with the single word: ready"}])
        print("  ✅ vision model responds")
    except LlmError as exc:
        print(f"  ⏳ vision model: {exc.kind}")
    if cfg["archive_dir"]:
        conn = _db(cfg)
        n = conn.execute("SELECT COUNT(*) c FROM media").fetchone()["c"]
        print(f"  🗂️  archive: {n} media files indexed at {cfg['archive_dir']}")
        conn.close()
    else:
        print("  ⏳ no archive configured — `alebrije init <directory>`")
    return 0 if ok else 1


def cmd_selftest(cfg, args) -> int:
    return selftest.run()


def cmd_update(cfg, args) -> int:
    from . import __version__
    latest = update.latest_release()
    if latest is None:
        print("⚠️  couldn't reach GitHub releases (offline? rate-limited?) — "
              "try again later")
        return 1
    if latest["tag"] is None:
        print("🕊️  no releases published yet — updates become available once "
              "a vX.Y.Z release is tagged on GitHub")
        return 0
    print(f"🌼 installed: v{__version__}   latest release: {latest['tag']}")
    if not update.is_newer(latest["tag"], __version__):
        print("✅ you're up to date")
        return 0
    if args.check:
        print(update.manual_instructions(latest["tag"]))
        return 0
    if update.pipx_managed():
        print(f"⬆️  upgrading via pipx from {update.install_source(latest['tag'])}…")
        if update.upgrade_with_pipx(latest["tag"]):
            print(f"✅ upgraded to {latest['tag']}")
            return 0
        print("❌ pipx upgrade failed — run it manually:")
        print(update.manual_instructions(latest["tag"]))
        return 1
    print(update.manual_instructions(latest["tag"]))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="alebrije",
        description="A local, open-weight AI spirit guide for your media archive.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("init", help="point Alebrije at your messy archive")
    sp.add_argument("archive", nargs="?", help="directory of photos/videos/audio")
    sp.add_argument("--home", help="state directory (default ~/.alebrije)")
    sp.add_argument("--vlm-url"), sp.add_argument("--embed-url"), sp.add_argument("--port", type=int)
    sp.set_defaults(func=cmd_init)

    sp = sub.add_parser("setup", help="install everything: llama.cpp + models (auto GPU)")
    sp.set_defaults(func=cmd_setup)

    sp = sub.add_parser("up", help="start the local model servers")
    sp.add_argument("--wait", type=float, default=600, help="seconds to wait for health")
    sp.set_defaults(func=cmd_up)

    sp = sub.add_parser("stop", help="stop the local model servers")
    sp.set_defaults(func=cmd_stop)

    sp = sub.add_parser("scan", help="index the archive directory (EXIF dates + GPS)")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("tag", help="caption + tag untagged media with the vision model")
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_tag)

    sp = sub.add_parser("name", help="match people-book names across tagged media")
    sp.add_argument("--limit", type=int, default=0)
    sp.set_defaults(func=cmd_name)

    sp = sub.add_parser("places", help="reverse-geocode any GPS-tagged media")
    sp.add_argument("--limit", type=int, default=25)
    sp.set_defaults(func=cmd_places)

    sp = sub.add_parser("person", help="manage the people book")
    sp.add_argument("which", choices=["add", "list", "rename"])
    sp.add_argument("name", nargs="?")
    sp.add_argument("new", nargs="?")
    sp.add_argument("--photo")
    sp.add_argument("--note", help="describe the person yourself instead of a photo")
    sp.set_defaults(func=cmd_person)

    sp = sub.add_parser("search", help="natural-language search over memories")
    sp.add_argument("query", nargs="*", default=[])
    sp.add_argument("--sort", choices=["relevance", "date", "type", "name"], default="relevance")
    sp.add_argument("--type", choices=["image", "video", "audio"])
    sp.add_argument("--person")
    sp.add_argument("--limit", type=int, default=60)
    sp.set_defaults(func=cmd_search)

    sp = sub.add_parser("serve", help="serve the web UI on your LAN")
    sp.add_argument("--host", default=None)
    sp.add_argument("--port", type=int, default=None)
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("doctor", help="check every piece of the machine")
    sp.set_defaults(func=cmd_doctor)

    sp = sub.add_parser("selftest", help="run the offline test suite")
    sp.set_defaults(func=cmd_selftest)

    sp = sub.add_parser("update", help="check GitHub releases and upgrade")
    sp.add_argument("--check", action="store_true",
                    help="only report; never reinstall")
    sp.set_defaults(func=cmd_update)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = config.load()
    return args.func(cfg, args)


if __name__ == "__main__":
    sys.exit(main())
