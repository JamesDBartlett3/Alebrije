---
title: I gave my messy photo archive a spirit guide — an offline, open-weight AI that learns your family, names your places, and finds any memory in plain language
published: true
tags: devchallenge, hf26challenge, python, ai
---

*This is a submission for the [Hacktoberfest Open-Source AI Challenge Week 4: Ghost in the Machine](https://dev.to/challenges/hacktoberfest-week4-2026-10-26)*

<!-- TODO before publishing:
  - record/attach a demo video or GIF in the Demo section (optional)
  - save your session with DevRelay and embed it in My Agent Session
-->

## What I Built

**Alebrije** ([github.com/JamesDBartlett3/Alebrije](https://github.com/JamesDBartlett3/Alebrije)) is a local-first archive spirit-guide for your photos, videos, and audio clips. Alebrijes are the brightly colored creatures of Mexican folk art that guide souls where they need to go — this one guides *you* through thousands of memories that would otherwise sit in folders with names like `DCIM_0947`.

You point it at a messy directory. It then:

1. **Meets your people.** You introduce a friend or family member from a sample photo (`alebrije person add Abuela --photo abuela.jpg`), and Alebrije builds a private *people book*: the face is embedded locally by open face models (SCRFD detection + AdaFace recognition via onnxruntime) and described in words by the vision model — no cloud, no face database handed to a third party.
2. **Recognizes them everywhere, and names every place.** Every photo is scanned for faces; each becomes a 512-d vector stored locally. Faces close to someone in your people book are matched automatically, borderline ones are double-checked by Gemma, and recurring strangers cluster into "Unnamed" people you name with one rename. It also tags every photo and video with objects, clothing, animals, food, and the *mood* of the moment. If a photo carries GPS coordinates, it finds the **closest point of interest** (a park, museum, landmark — nearest by real haversine distance, via the OpenStreetMap Overpass API) and **always appends the municipality chain**: `Civic Center Park — Denver, Colorado, USA`, or with no landmark in range simply `City of Golden, Jefferson County, Colorado, USA`.
3. **Remembers by feeling, not filename.** Everything is embedded with an open model into a local SQLite index, so you can search in plain language — *"dog on the beach at sunset"* — sorted by relevance, newest, media type, or name.

Everything runs on your machine: an open-weight multimodal model (Gemma 3 4B via llama.cpp) looks at your media, open face models (SCRFD + AdaFace via onnxruntime) recognize your people, an open embedding model (nomic-embed-text) indexes it, and a Día de los Muertos–themed local web app — papel picado banner, marigold glow — serves the whole collection to any phone or laptop on your network. Nothing about your family ever leaves your hardware (the only outbound calls, and only for GPS-tagged files, are the two open-data OpenStreetMap geocoders, which receive coordinates, never images).

I built this for the #GhostInTheMachine theme the way Día de los Muertos itself treats it: the dead stay alive as long as their stories are told. A hard drive full of untagged photos is a room full of ancestors waiting to be introduced. Alebrije is the marigold path through it.

## Demo

Everything below is from a live run on an ordinary 4-core desktop, CPU only:

![Alebrije web gallery](https://raw.githubusercontent.com/JamesDBartlett3/Alebrije/main/screenshots/web-desktop.png)

Six files went in named `IMG_4031.png` … `IMG_4043.jpg`. Alebrije tagged all six (captions, objects, clothing, mood), recognized and named a person across four of them from a single sample photo, and reverse-geocoded the two that carried GPS. Then, in plain language:

![Natural-language search](https://raw.githubusercontent.com/JamesDBartlett3/Alebrije/main/screenshots/web-search.png)

`"dog on the beach at sunset"` puts the three beach-dog photos on top — including one taken before the archive's newest party photos — because ranking is by meaning, not date.

![Memory detail](https://raw.githubusercontent.com/JamesDBartlett3/Alebrije/main/screenshots/web-detail.png)

The CLI speaks the same language:

![CLI search](https://raw.githubusercontent.com/JamesDBartlett3/Alebrije/main/screenshots/cli-search.png)

And it's the same app on a phone, served from your own desk over the LAN:

![Phone view](https://raw.githubusercontent.com/JamesDBartlett3/Alebrije/main/screenshots/web-phone.png)

<!-- TODO: optional — record a short video walk-through and attach it here -->

## Code

https://github.com/JamesDBartlett3/Alebrije

Install and run (Python 3.10+ and [pipx](https://pipx.pypa.io/) are the only prerequisites — Alebrije installs everything else):

```bash
pipx install git+https://github.com/JamesDBartlett3/Alebrije.git@v0.2.0
alebrije setup        # downloads llama.cpp + the open-weight models, auto-detects your GPU
alebrije init ~/Pictures/family
alebrije up           # starts the two local model servers
alebrije scan         # index files, read EXIF dates + GPS
alebrije tag          # vision-tag everything (CPU: ~1 min/photo, GPU: seconds)
alebrije serve        # http://localhost:8799 — also on your LAN
alebrije update       # later: checks GitHub releases, upgrades in place
```

No pipx? `git clone` + `python alebrije.py <command>` works straight from the repo — friendliest option on Windows.

## How I Built It

- **Vision + language:** [Gemma 3 4B IT](https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF) (open weights, Gemma license) served by **llama.cpp's** `llama-server` with its multimodal projector. Every caption, object list, clothing item, animal, mood, person description, and gray-band identity double-check comes from this one local model.
- **Embeddings & search:** [nomic-embed-text v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF) (Apache-2.0), via llama.cpp's OpenAI-compatible embeddings endpoint. Vectors live as float32 blobs in SQLite; cosine ranking is a few lines of Python.
- **Faces, without the cloud:** v0.1 matched people by Gemma descriptions alone — we shipped it, watched it conflate lookalike relatives and miss people across decades, measured the failure, and rebuilt identity on real face embeddings. [SCRFD-10GF](https://github.com/deepinsight/insightface) detects every face (five landmarks each) and [AdaFace ir101](https://github.com/yakhyo/adaface-onnx) — trained specifically to handle *low-quality* faces, i.e. old family photos — turns each aligned crop into a 512-d identity vector, both on onnxruntime (MIT) on CPU. Detection runs a letterboxed whole-image pass *plus* overlapping native-resolution tiles on large photos, so faces never get lost to downsampling; alignment crops are always warped from the full-resolution original. Matching uses measured thresholds (same person ≈ 0.56–0.95, different people ≈ 0.0), with Gemma confirming borderline cases and strangers auto-clustering into nameable "Unnamed" people.
- **The app itself is nearly all Python standard library** — a scanner (with a hand-rolled EXIF parser that pulls capture dates *and* GPS coordinates out of JPEG APP1 segments), a tagging pipeline, a people book, a vector index, a CLI, and a threaded web server with an embedded single-page UI. The only pip dependencies are `onnxruntime` and `Pillow`. No framework, no account.
- **Fully self-installing, fully cross-platform.** `alebrije setup` works on Linux, macOS, and Windows: it queries the llama.cpp GitHub releases API and downloads the prebuilt build matching your OS, CPU architecture, and **accelerator** — NVIDIA (`nvidia-smi`), AMD ROCm (`rocm-smi`), Vulkan (`vulkaninfo`), Apple Metal on any Mac, or plain CPU — then fetches the five open models (three GGUFs from Hugging Face, two face ONNXs from GitHub releases) with resume support. When a GPU is present, `alebrije up` launches the models with full layer offload (`-ngl 999`).
- **Tests:** 182 offline tests (`alebrije selftest`) cover the scanner (including a synthetic JPEG with hand-built EXIF + GPS binary), the face pipeline (letterbox/tiling math, SCRFD decoding, similarity alignment, threshold bands, clustering, v0.1-database migration), tag coercion, people matching, vector math, search sorting, the bootstrap asset picker for every OS/GPU combo, server lifecycle, the web API (real HTTP on an ephemeral port), and the CLI — all against fake model servers, no network or GPU required.

## Why Does Open Innovation Matter?

Your family photos are the last data you'd ever hand to a stranger, and the first data that gets uploaded without thinking the moment you use a cloud "AI photo assistant." Open weights flip that: the same Gemma model a giant cloud product would run is running *on your desk*, inspectable, with your files never leaving the room.

Openness also made this project possible at all. llama.cpp's OpenAI-compatible server meant the whole app is one `urllib` call away from vision and embeddings; the GGUF ecosystem meant a 4B model runs on an ordinary 4-core CPU; permissive licenses meant everything here ships without negotiation; and OpenStreetMap's open geodata meant place names without a paid maps API. If the models were closed, an offline family-archive assistant simply couldn't exist — and if they were free-but-closed, it couldn't be audited, fine-tuned, or trusted with something as personal as a memorial archive.

## My Agent Session

<!-- TODO: save your session with DevRelay and embed it here with the
     agent_session tag (see the challenge page), or paste the session link -->

## The Spirit of It

The moment it landed: I fed it six files named `IMG_40xx` and asked, out loud, *"dog on the beach at sunset"* — and the first three results were exactly the three photos I'd have picked, one of them taken years before the others. A folder became a memory. That's the whole ofrenda: the tech is just the marigold petals; what they guide home is your own past.

## Prize Categories

- **Gemma** (Featured) — Alebrije is built around Gemma 3 4B: every caption, tag, person description, and borderline identity double-check comes from the open-weight Gemma model running locally through llama.cpp.

<!-- Other partner categories (Render, TabPFN, Tinker, Arduino, DigitalOcean,
     Auth0, Backboard, ElevenLabs, Browserbase, Entire, GitHub Copilot,
     Mastra, MongoDB Atlas, Ultramarine, Sentry, SerpApi, .tech, Temporal,
     Tiger Data) — add any others that apply. -->

<!-- Team Submissions: single-author submission. -->

<!-- Thanks for participating! -->
