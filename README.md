<p align="center">
  <img src="logo.png" alt="Egg Run" width="128" />
</p>

<h1 align="center">Egg LLM</h1>

<p align="center">
  Real, GPU-accelerated LLM inference <b>inside a hardware-isolated VM</b>, on the
  host <b>Apple GPU</b> — driven from inside the egg.
</p>

## Where this runs — [egg](https://github.com/EggRunAI/egg-downloads)

This VM runs under **egg** (the EggRun hypervisor): Apple **Hypervisor.framework**,
in **userspace**, **no daemon**. An "egg" is a hardware-isolated VM — EggRun is a
Docker competitor that uses VMs instead of a shared kernel: *Containers share a
kernel. Eggs don't.*

- **Guest:** aarch64 Ubuntu (this VM).
- **Host:** Apple Silicon (M5 Max), macOS, running `egg`.
- **GPU:** the real host Apple GPU, reached from the guest over **virtio-gpu**.

## Proven

Qwen2.5-0.5B-Instruct (Q4_K_M), `-ngl 99` → **real, coherent tokens at ~295–350
tok/s** on the Apple M5 Max GPU, generated from inside the egg — with the GNOME
desktop still rendering on Venus *alongside* it (APIR capset 10 coexists with the
graphics path; the GUI never flinches).

## Run it (in the guest)

```bash
./build.sh    # install dependencies and build
./test.sh     # test the build
./egg-llm     # launch the TUI and chat with the model
```

Keep the model in `~` (persistent) — `/tmp` is wiped on reboot.

---

**EggRun** — hardware-isolated VMs for autonomous workloads. eggrun.ai · a
Camouflage Networks company.