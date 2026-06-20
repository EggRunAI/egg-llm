//! egg-llm — interactive, streaming chat for in-VM GPU LLM inference.
//!
//! Same engine as `llm-infer`: links `libllama` (the prebuilt Vulkan build) via
//! FFI and runs the model in-process with every layer on the GPU. This binary
//! adds a multi-turn chat loop that **streams tokens live** as they generate.
//!
//! Pure std — no external crates. Raw terminal handling is done with `stty`
//! (coreutils), so there is no fragile `termios` struct to transcribe.
//!
//! Build (in the guest):  cargo build --release --bin egg-llm
//! Run:                   LLM_MODEL=~/models/model.gguf ./target/release/egg-llm

use std::ffi::{c_char, c_int, c_void, CString};
use std::io::{Read, Write};
use std::time::Instant;

// ---- opaque handles (identical to src/main.rs) ----
type Model = c_void;
type Context = c_void;
type Vocab = c_void;
type Sampler = c_void;
type LlamaToken = i32;

// ---- structs (transcribed from vendor/llama.h @ b9670, same as src/main.rs) ----
#[repr(C)]
struct ModelParams {
    devices: *mut c_void,
    tensor_buft_overrides: *const c_void,
    n_gpu_layers: i32,
    split_mode: i32,
    main_gpu: i32,
    tensor_split: *const f32,
    progress_callback: *mut c_void,
    progress_callback_user_data: *mut c_void,
    kv_overrides: *const c_void,
    vocab_only: bool,
    use_mmap: bool,
    use_direct_io: bool,
    use_mlock: bool,
    check_tensors: bool,
    use_extra_bufts: bool,
    no_host: bool,
    no_alloc: bool,
}

#[repr(C)]
struct ContextParams {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_rs_seq: u32,
    n_outputs_max: u32,
    n_threads: i32,
    n_threads_batch: i32,
    ctx_type: i32,
    rope_scaling_type: i32,
    pooling_type: i32,
    attention_type: i32,
    flash_attn_type: i32,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    defrag_thold: f32,
    cb_eval: *mut c_void,
    cb_eval_user_data: *mut c_void,
    type_k: i32,
    type_v: i32,
    abort_callback: *mut c_void,
    abort_callback_data: *mut c_void,
    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    samplers: *mut c_void,
    n_samplers: usize,
    ctx_other: *mut c_void,
}

#[repr(C)]
struct SamplerChainParams {
    no_perf: bool,
}

#[repr(C)]
struct Batch {
    n_tokens: i32,
    token: *mut LlamaToken,
    embd: *mut f32,
    pos: *mut i32,
    n_seq_id: *mut i32,
    seq_id: *mut *mut i32,
    logits: *mut i8,
}

#[repr(C)]
struct ChatMessage {
    role: *const c_char,
    content: *const c_char,
}

#[link(name = "llama")]
extern "C" {
    fn ggml_backend_load_all_from_path(dir_path: *const c_char);
    fn llama_backend_init();
    fn llama_backend_free();

    fn llama_model_default_params() -> ModelParams;
    fn llama_context_default_params() -> ContextParams;
    fn llama_sampler_chain_default_params() -> SamplerChainParams;

    fn llama_model_load_from_file(path: *const c_char, params: ModelParams) -> *mut Model;
    fn llama_model_free(model: *mut Model);
    fn llama_init_from_model(model: *mut Model, params: ContextParams) -> *mut Context;
    fn llama_free(ctx: *mut Context);

    fn llama_model_get_vocab(model: *const Model) -> *const Vocab;
    fn llama_n_ctx(ctx: *const Context) -> u32;
    fn llama_model_chat_template(model: *const Model, name: *const c_char) -> *const c_char;

    fn llama_chat_apply_template(
        tmpl: *const c_char,
        chat: *const ChatMessage,
        n_msg: usize,
        add_ass: bool,
        buf: *mut c_char,
        length: c_int,
    ) -> c_int;

    fn llama_tokenize(
        vocab: *const Vocab,
        text: *const c_char,
        text_len: c_int,
        tokens: *mut LlamaToken,
        n_tokens_max: c_int,
        add_special: bool,
        parse_special: bool,
    ) -> c_int;

    fn llama_token_to_piece(
        vocab: *const Vocab,
        token: LlamaToken,
        buf: *mut c_char,
        length: c_int,
        lstrip: c_int,
        special: bool,
    ) -> c_int;

    fn llama_batch_get_one(tokens: *mut LlamaToken, n_tokens: c_int) -> Batch;
    fn llama_decode(ctx: *mut Context, batch: Batch) -> c_int;
    fn llama_vocab_is_eog(vocab: *const Vocab, token: LlamaToken) -> bool;

    fn llama_sampler_chain_init(params: SamplerChainParams) -> *mut Sampler;
    fn llama_sampler_chain_add(chain: *mut Sampler, smpl: *mut Sampler);
    fn llama_sampler_free(smpl: *mut Sampler);
    fn llama_sampler_init_top_k(k: i32) -> *mut Sampler;
    fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut Sampler;
    fn llama_sampler_init_temp(t: f32) -> *mut Sampler;
    fn llama_sampler_init_dist(seed: u32) -> *mut Sampler;
    fn llama_sampler_sample(smpl: *mut Sampler, ctx: *mut Context, idx: c_int) -> LlamaToken;
}

// ---- tiny ANSI helpers ----
const DIM: &str = "\x1b[2m";
const BOLD: &str = "\x1b[1m";
const GOLD: &str = "\x1b[38;5;220m"; // yolk
const RESET: &str = "\x1b[0m";

fn flush() {
    std::io::stdout().flush().ok();
}

/// Put the terminal in raw mode via `stty` and restore it on drop. Avoids
/// transcribing the platform `termios` struct.
struct RawTty {
    saved: Option<String>,
}
impl RawTty {
    fn enable() -> RawTty {
        let saved = std::process::Command::new("stty")
            .arg("-g")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
        // -echo: don't echo input; -icanon: deliver bytes immediately.
        let _ = std::process::Command::new("stty").args(["-echo", "-icanon"]).status();
        RawTty { saved }
    }
}
impl Drop for RawTty {
    fn drop(&mut self) {
        match &self.saved {
            Some(s) if !s.is_empty() => {
                let _ = std::process::Command::new("stty").arg(s).status();
            }
            _ => {
                let _ = std::process::Command::new("stty").arg("sane").status();
            }
        }
        println!("{RESET}");
    }
}

/// Download a default model with `curl` if the path doesn't exist yet — the FFI
/// engine can't fetch one itself. Override the URL with LLM_MODEL_URL, or point
/// LLM_MODEL at a local file to skip the download entirely.
fn ensure_model(path: &str) {
    if std::path::Path::new(path).exists() {
        return;
    }
    let url = std::env::var("LLM_MODEL_URL").unwrap_or_else(|_| {
        "https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf".to_string()
    });
    println!("{DIM}  model '{path}' not found — downloading…{RESET}");
    if let Some(parent) = std::path::Path::new(path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let ok = std::process::Command::new("curl")
        .args(["-L", "--fail", "--retry", "3", "-o", path, &url])
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        eprintln!("error: could not download a model. Set LLM_MODEL to a local .gguf, or LLM_MODEL_URL to a URL.");
        std::process::exit(1);
    }
    println!("{DIM}  model ready → {path}{RESET}\n");
}

fn main() {
    // Select the APIR capset — the host GPU via API Remoting. Without this the
    // ggml-virtgpu frontend defaults to the Venus capset (the graphics path),
    // whose handshake times out for compute and the process aborts. Equivalent
    // to running with GGML_REMOTING_USE_APIR_CAPSET=1 (what llama-cli used).
    std::env::set_var("GGML_REMOTING_USE_APIR_CAPSET", "1");

    let model_path = std::env::var("LLM_MODEL").unwrap_or_else(|_| "models/model.gguf".into());
    let n_gpu_layers: i32 = std::env::var("LLM_NGL").ok().and_then(|v| v.parse().ok()).unwrap_or(99);
    let n_ctx: u32 = std::env::var("LLM_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(8192);

    print!("\x1b[2J\x1b[H");
    println!("{GOLD}{BOLD}  Egg LLM{RESET}{DIM} — streaming chat, in-VM on the host GPU{RESET}");
    println!("{DIM}  model {model_path}  ·  ngl {n_gpu_layers}  ·  ctx {n_ctx}{RESET}");
    println!("{DIM}  type and press Enter · /reset clears the chat · /quit (or Ctrl-D) exits{RESET}\n");

    ensure_model(&model_path);

    unsafe {
        let lib_dir = std::env::var("LLM_LIB_DIR")
            .unwrap_or_else(|_| env!("LLM_LIB_DIR_DEFAULT").to_string());
        let lib_dir_c = CString::new(lib_dir).unwrap();
        ggml_backend_load_all_from_path(lib_dir_c.as_ptr());
        llama_backend_init();

        let mp = CString::new(model_path.as_str()).unwrap();
        let mut mparams = llama_model_default_params();
        mparams.n_gpu_layers = n_gpu_layers;
        let t_load = Instant::now();
        let model = llama_model_load_from_file(mp.as_ptr(), mparams);
        if model.is_null() {
            eprintln!("error: failed to load model '{model_path}'");
            std::process::exit(1);
        }
        println!("{DIM}  model loaded in {:.0} ms{RESET}\n", t_load.elapsed().as_secs_f64() * 1000.0);
        let vocab = llama_model_get_vocab(model);

        // History of (role, content). Re-templated each turn against a fresh
        // context — correct and simple; incremental KV reuse is a later optimization.
        let mut history: Vec<(String, String)> = Vec::new();

        loop {
            print!("{GOLD}{BOLD}you ›{RESET} ");
            flush();
            let line = match read_line() {
                Some(l) => l,
                None => {
                    println!("\n{DIM}bye{RESET}");
                    break;
                }
            };
            let user = line.trim();
            if user.is_empty() {
                continue;
            }
            match user {
                "/quit" | "/exit" => break,
                "/reset" => {
                    history.clear();
                    println!("{DIM}  (chat reset){RESET}\n");
                    continue;
                }
                _ => {}
            }
            history.push(("user".into(), user.to_string()));

            // Build the full prompt from history via the model's chat template.
            let prompt = apply_chat_template(model, &history);
            let mut toks = tokenize(vocab, &prompt, true);
            if toks.is_empty() {
                println!("{DIM}  (empty prompt){RESET}");
                history.pop();
                continue;
            }

            // Fresh context per turn so the KV cache never carries stale state.
            let mut cparams = llama_context_default_params();
            cparams.n_ctx = n_ctx;
            cparams.n_batch = n_ctx;
            let nthreads = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4);
            cparams.n_threads = nthreads;
            cparams.n_threads_batch = nthreads;
            let ctx = llama_init_from_model(model, cparams);
            if ctx.is_null() {
                eprintln!("error: failed to create context");
                break;
            }
            let ctx_cap = llama_n_ctx(ctx) as i32;

            let smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
            llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95, 1));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7));
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(0xC0FFEE));

            print!("{GOLD}{BOLD}egg ›{RESET} ");
            flush();

            let mut reply = String::new();
            let mut n_decoded = 0i32;
            let mut n_past = 0i32;
            let mut cur = toks.clone();
            let n_prompt = cur.len() as i32;
            toks.clear();
            // The first decode processes the whole prompt (prefill); every later
            // decode is a single generated token. Time the two phases separately
            // so prompt throughput doesn't get folded into the decode rate.
            let mut prefill_s = 0.0f64;
            let mut t_decode = Instant::now();
            loop {
                if n_past + cur.len() as i32 > ctx_cap {
                    print!("{DIM}[context full]{RESET}");
                    break;
                }
                let is_prefill = n_past == 0;
                let t_step = Instant::now();
                let batch = llama_batch_get_one(cur.as_mut_ptr(), cur.len() as c_int);
                if llama_decode(ctx, batch) != 0 {
                    eprintln!("\nerror: llama_decode failed");
                    break;
                }
                n_past += cur.len() as i32;
                if is_prefill {
                    prefill_s = t_step.elapsed().as_secs_f64();
                    t_decode = Instant::now();
                }

                let tok = llama_sampler_sample(smpl, ctx, -1);
                if llama_vocab_is_eog(vocab, tok) {
                    break;
                }
                let piece = token_to_piece(vocab, tok);
                let s = String::from_utf8_lossy(&piece);
                print!("{s}"); // stream live
                flush();
                reply.push_str(&s);
                n_decoded += 1;
                if n_decoded >= 4096 {
                    break;
                }
                cur = vec![tok];
            }
            let decode_s = t_decode.elapsed().as_secs_f64();
            let prompt_tps = if prefill_s > 0.0 { n_prompt as f64 / prefill_s } else { 0.0 };
            let decode_tps = if decode_s > 0.0 { n_decoded as f64 / decode_s } else { 0.0 };
            println!(
                "\n{DIM}  prompt {n_prompt} tok @ {prompt_tps:.1} t/s · decode {n_decoded} tok @ {decode_tps:.1} t/s{RESET}\n"
            );

            history.push(("assistant".into(), reply.trim().to_string()));

            llama_sampler_free(smpl);
            llama_free(ctx);
        }

        llama_model_free(model);
        llama_backend_free();
    }
}

/// Read one line from stdin in raw mode, echoing printable chars and handling
/// Backspace. Returns None on EOF (Ctrl-D) or read error.
fn read_line() -> Option<String> {
    let _raw = RawTty::enable();
    let mut stdin = std::io::stdin().lock();
    let mut buf = String::new();
    let mut byte = [0u8; 1];
    loop {
        if stdin.read(&mut byte).ok()? == 0 {
            return if buf.is_empty() { None } else { Some(buf) };
        }
        match byte[0] {
            b'\n' | b'\r' => {
                println!();
                return Some(buf);
            }
            4 => {
                // Ctrl-D
                return if buf.is_empty() { None } else { Some(buf) };
            }
            3 => {
                // Ctrl-C
                println!();
                std::process::exit(0);
            }
            8 | 127 => {
                // Backspace / DEL
                if buf.pop().is_some() {
                    print!("\x08 \x08");
                    flush();
                }
            }
            b if b >= 0x20 => {
                let c = b as char;
                buf.push(c);
                print!("{c}");
                flush();
            }
            _ => {}
        }
    }
}

/// Format the whole conversation using the model's built-in chat template.
unsafe fn apply_chat_template(model: *const Model, history: &[(String, String)]) -> String {
    let tmpl = llama_model_chat_template(model, std::ptr::null());

    // Keep the CStrings alive for the duration of the FFI call.
    let cstrs: Vec<(CString, CString)> = history
        .iter()
        .map(|(r, c)| (CString::new(r.as_str()).unwrap(), CString::new(c.as_str()).unwrap()))
        .collect();
    let msgs: Vec<ChatMessage> = cstrs
        .iter()
        .map(|(r, c)| ChatMessage { role: r.as_ptr(), content: c.as_ptr() })
        .collect();

    if !tmpl.is_null() {
        let total: usize = history.iter().map(|(_, c)| c.len()).sum::<usize>() + 1024;
        let mut buf = vec![0i8; total];
        let mut n = llama_chat_apply_template(
            tmpl,
            msgs.as_ptr(),
            msgs.len(),
            true,
            buf.as_mut_ptr() as *mut c_char,
            buf.len() as c_int,
        );
        if n > buf.len() as c_int {
            buf = vec![0i8; n as usize];
            n = llama_chat_apply_template(
                tmpl,
                msgs.as_ptr(),
                msgs.len(),
                true,
                buf.as_mut_ptr() as *mut c_char,
                buf.len() as c_int,
            );
        }
        if n > 0 {
            let bytes: Vec<u8> = buf[..n as usize].iter().map(|&b| b as u8).collect();
            return String::from_utf8_lossy(&bytes).into_owned();
        }
    }

    // Fallback: ChatML (what Qwen uses) if no template is embedded.
    let mut s = String::new();
    for (role, content) in history {
        s.push_str(&format!("<|im_start|>{role}\n{content}<|im_end|>\n"));
    }
    s.push_str("<|im_start|>assistant\n");
    s
}

unsafe fn tokenize(vocab: *const Vocab, text: &str, add_special: bool) -> Vec<LlamaToken> {
    let c = CString::new(text).unwrap();
    let len = text.len() as c_int;
    let n = llama_tokenize(vocab, c.as_ptr(), len, std::ptr::null_mut(), 0, add_special, true);
    let need = (-n).max(0) as usize;
    if need == 0 {
        return Vec::new();
    }
    let mut toks = vec![0i32; need];
    let got = llama_tokenize(vocab, c.as_ptr(), len, toks.as_mut_ptr(), need as c_int, add_special, true);
    toks.truncate(got.max(0) as usize);
    toks
}

unsafe fn token_to_piece(vocab: *const Vocab, token: LlamaToken) -> Vec<u8> {
    let mut buf = [0i8; 256];
    let n = llama_token_to_piece(vocab, token, buf.as_mut_ptr() as *mut c_char, buf.len() as c_int, 0, false);
    if n <= 0 {
        return Vec::new();
    }
    buf[..n as usize].iter().map(|&b| b as u8).collect()
}