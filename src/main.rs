//! llm-infer — in-process GPU LLM inference on the Apple M5 Max, in Rust.
//!
//! No server, no HTTP. This binary links `libllama.so` (the prebuilt llama.cpp
//! Vulkan build) directly via FFI and runs the model inside the process. With
//! `n_gpu_layers = 99` every transformer layer is offloaded to the GPU, which on
//! this VM is the Apple M5 Max exposed through Virtio-GPU Venus / Vulkan.
//!
//! This milestone: take a prompt, run it through the model's chat template,
//! generate the full reply, and print it once (no streaming yet). A TUI comes next.

use std::ffi::{c_char, c_int, c_void, CString};
use std::time::Instant;

// ---- opaque handles ----
type Model = c_void;
type Context = c_void;
type Vocab = c_void;
type Sampler = c_void;
type Memory = c_void;
type LlamaToken = i32;

// ---- structs (transcribed exactly from vendor/llama.h @ b9670) ----
// Enums are C `int` (4 bytes) -> i32. Pointers/callbacks are 8 bytes.

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
    fn llama_vocab_n_tokens(vocab: *const Vocab) -> i32;
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
    fn llama_sampler_init_greedy() -> *mut Sampler;
    fn llama_sampler_init_top_k(k: i32) -> *mut Sampler;
    fn llama_sampler_init_top_p(p: f32, min_keep: usize) -> *mut Sampler;
    fn llama_sampler_init_temp(t: f32) -> *mut Sampler;
    fn llama_sampler_init_dist(seed: u32) -> *mut Sampler;
    fn llama_sampler_sample(smpl: *mut Sampler, ctx: *mut Context, idx: c_int) -> LlamaToken;

    #[allow(dead_code)]
    fn llama_get_memory(ctx: *const Context) -> *mut Memory;
    #[allow(dead_code)]
    fn llama_memory_clear(mem: *mut Memory, data: bool);
}

struct Args {
    model: String,
    prompt: String,
    n_gpu_layers: i32,
    n_ctx: u32,
    max_tokens: i32,
    self_test: bool,
}

fn parse_args() -> Args {
    let mut a = Args {
        model: std::env::var("LLM_MODEL").unwrap_or_else(|_| "models/model.gguf".into()),
        prompt: String::new(),
        n_gpu_layers: 99,
        n_ctx: 4096,
        max_tokens: 256,
        self_test: false,
    };
    let mut it = std::env::args().skip(1);
    let mut parts = Vec::new();
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "-m" | "--model" => a.model = it.next().unwrap_or(a.model),
            "-p" | "--prompt" => {
                if let Some(p) = it.next() {
                    parts.push(p)
                }
            }
            "-n" | "--max-tokens" => {
                a.max_tokens = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.max_tokens)
            }
            "--ngl" => a.n_gpu_layers = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.n_gpu_layers),
            "--ctx" => a.n_ctx = it.next().and_then(|v| v.parse().ok()).unwrap_or(a.n_ctx),
            "--self-test" => a.self_test = true,
            _ => parts.push(arg),
        }
    }
    if !parts.is_empty() {
        a.prompt = parts.join(" ");
    }
    if a.prompt.is_empty() {
        a.prompt = "In two sentences, explain what makes GPU inference fast.".into();
    }
    a
}

fn main() {
    let args = parse_args();

    eprintln!("╭─ llm-infer (in-process FFI → libllama → Vulkan/M5 Max)");
    eprintln!("│  model      : {}", args.model);
    eprintln!("│  gpu layers : {}", args.n_gpu_layers);
    eprintln!("│  context    : {}", args.n_ctx);
    eprintln!("╰─ prompt     : {:?}", args.prompt);

    unsafe {
        // Register the ggml backends (incl. Vulkan) from the prebuilt lib dir,
        // then init llama. The default loader searches near the exe/CWD, so we
        // give it the explicit directory where libggml-vulkan.so lives.
        let lib_dir = std::env::var("LLM_LIB_DIR")
            .unwrap_or_else(|_| env!("LLM_LIB_DIR_DEFAULT").to_string());
        let lib_dir_c = CString::new(lib_dir.clone()).unwrap();
        ggml_backend_load_all_from_path(lib_dir_c.as_ptr());
        llama_backend_init();

        // ---- load model on GPU ----
        let model_path = CString::new(args.model.as_str()).unwrap();
        let mut mparams = llama_model_default_params();
        mparams.n_gpu_layers = args.n_gpu_layers;

        let t_load = Instant::now();
        let model = llama_model_load_from_file(model_path.as_ptr(), mparams);
        if model.is_null() {
            eprintln!("error: failed to load model '{}'", args.model);
            std::process::exit(1);
        }
        let load_ms = t_load.elapsed().as_secs_f64() * 1000.0;
        eprintln!("✅ model loaded in {load_ms:.0} ms");

        let vocab = llama_model_get_vocab(model);

        // ---- backend correctness self-test ----
        // A working backend's next-token prediction depends on the prompt; a
        // miscomputing one (e.g. the Virtio-GPU Venus passthrough on this VM,
        // which returns constant garbage logits) predicts the SAME first token
        // no matter what you ask it. Probe with three prompts that have wildly
        // different correct continuations and assert the predictions diverge.
        if args.self_test {
            let ok = run_self_test(model, vocab, args.n_gpu_layers);
            llama_model_free(model);
            llama_backend_free();
            std::process::exit(if ok { 0 } else { 2 });
        }

        // ---- context ----
        let mut cparams = llama_context_default_params();
        cparams.n_ctx = args.n_ctx;
        cparams.n_batch = args.n_ctx;
        let nthreads = std::thread::available_parallelism().map(|n| n.get() as i32).unwrap_or(4);
        cparams.n_threads = nthreads;
        cparams.n_threads_batch = nthreads;
        let ctx = llama_init_from_model(model, cparams);
        if ctx.is_null() {
            eprintln!("error: failed to create context");
            std::process::exit(1);
        }
        let n_ctx = llama_n_ctx(ctx) as i32;

        // ---- sampler chain: top_k -> top_p -> temp -> dist ----
        let smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        llama_sampler_chain_add(smpl, llama_sampler_init_top_k(40));
        llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.95, 1));
        llama_sampler_chain_add(smpl, llama_sampler_init_temp(0.7));
        llama_sampler_chain_add(smpl, llama_sampler_init_dist(0xC0FFEE));

        // ---- build prompt via the model's chat template ----
        let formatted = apply_chat_template(model, &args.prompt);
        let mut tokens = tokenize(vocab, &formatted, true);
        if tokens.is_empty() {
            eprintln!("error: prompt tokenized to nothing");
            std::process::exit(1);
        }

        eprintln!("⏳ generating ({} prompt tokens)…\n", tokens.len());

        // ---- decode loop (non-streaming: accumulate, print at end) ----
        let mut out_bytes: Vec<u8> = Vec::new();
        let mut n_decoded = 0i32;
        let mut n_past = 0i32;
        let mut cur = tokens.clone();
        tokens.clear();

        let t_gen = Instant::now();
        loop {
            if n_past + cur.len() as i32 > n_ctx {
                eprintln!("\n[context full]");
                break;
            }
            let batch = llama_batch_get_one(cur.as_mut_ptr(), cur.len() as c_int);
            if llama_decode(ctx, batch) != 0 {
                eprintln!("\nerror: llama_decode failed");
                break;
            }
            n_past += cur.len() as i32;

            let tok = llama_sampler_sample(smpl, ctx, -1);
            if llama_vocab_is_eog(vocab, tok) {
                break;
            }
            out_bytes.extend_from_slice(&token_to_piece(vocab, tok));
            n_decoded += 1;
            if n_decoded >= args.max_tokens {
                break;
            }
            cur = vec![tok];
        }
        let gen_s = t_gen.elapsed().as_secs_f64();

        // ---- print the full reply once ----
        let reply = String::from_utf8_lossy(&out_bytes);
        println!("{}", reply.trim());

        let tps = if gen_s > 0.0 { n_decoded as f64 / gen_s } else { 0.0 };
        eprintln!("\n╭─ done");
        eprintln!("│  generated  : {n_decoded} tokens in {gen_s:.2} s");
        eprintln!("│  throughput : {tps:.1} tok/s");
        eprintln!("╰─ backend    : Vulkan / Virtio-GPU Venus (Apple M5 Max)");

        // ---- cleanup ----
        llama_sampler_free(smpl);
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
    }
}

/// Greedily predict the first response token for `prompt` in a fresh context.
/// Returns the token id and its decoded text. A fresh context per probe keeps
/// the KV cache from leaking state between probes.
unsafe fn first_token(model: *mut Model, vocab: *const Vocab, prompt: &str) -> (LlamaToken, String) {
    let mut cparams = llama_context_default_params();
    cparams.n_ctx = 512;
    cparams.n_batch = 512;
    let ctx = llama_init_from_model(model, cparams);
    if ctx.is_null() {
        return (-1, String::from("<ctx-failed>"));
    }
    let smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    let formatted = apply_chat_template(model, prompt);
    let mut toks = tokenize(vocab, &formatted, true);
    let tok = if toks.is_empty() {
        -1
    } else {
        let batch = llama_batch_get_one(toks.as_mut_ptr(), toks.len() as c_int);
        if llama_decode(ctx, batch) != 0 {
            -1
        } else {
            llama_sampler_sample(smpl, ctx, -1)
        }
    };
    let piece = String::from_utf8_lossy(&token_to_piece(vocab, tok)).into_owned();

    llama_sampler_free(smpl);
    llama_free(ctx);
    (tok, piece)
}

/// Verify the active backend actually computes: greedily decode three prompts
/// with distinct correct answers and check the predicted first tokens diverge.
/// Returns true on PASS. See FOR_HYPERVISOR.md for why this catches the Venus
/// passthrough bug (constant, prompt-independent logits).
unsafe fn run_self_test(model: *mut Model, vocab: *const Vocab, n_gpu_layers: i32) -> bool {
    let probes = [
        "The capital of France is",
        "2 + 2 =",
        "Write a haiku about the ocean.",
    ];
    eprintln!("\n╭─ backend self-test (greedy, {} probes, ngl={})", probes.len(), n_gpu_layers);
    let mut results: Vec<(LlamaToken, String)> = Vec::new();
    for p in probes {
        let (tok, piece) = first_token(model, vocab, p);
        eprintln!("│  {:<34} → token {:>6}  {:?}", format!("{:?}", p), tok, piece);
        results.push((tok, piece));
    }

    let distinct: std::collections::BTreeSet<LlamaToken> = results.iter().map(|(t, _)| *t).collect();
    let pass = distinct.len() > 1 && !results.iter().any(|(t, _)| *t < 0);

    if pass {
        eprintln!("╰─ PASS ✅  predictions are prompt-dependent — backend computes correctly");
    } else if distinct.len() == 1 {
        eprintln!("╰─ FAIL ❌  identical prediction for every prompt — backend returns");
        eprintln!("            constant/garbage logits (the Venus/Vulkan passthrough bug).");
        eprintln!("            See FOR_HYPERVISOR.md. Workaround: GGML_VK_VISIBLE_DEVICES");
        eprintln!("            selecting llvmpipe, or run CPU-only without libggml-vulkan.so.");
    } else {
        eprintln!("╰─ FAIL ❌  decode/sampling error during probe (negative token id)");
    }
    pass
}

/// Format a single user turn using the model's built-in chat template.
unsafe fn apply_chat_template(model: *const Model, user: &str) -> String {
    let tmpl = llama_model_chat_template(model, std::ptr::null());
    let role = CString::new("user").unwrap();
    let content = CString::new(user).unwrap();
    let msgs = [ChatMessage {
        role: role.as_ptr(),
        content: content.as_ptr(),
    }];

    if !tmpl.is_null() {
        let mut buf = vec![0i8; user.len() + 512];
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
    format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n")
}

unsafe fn tokenize(vocab: *const Vocab, text: &str, add_special: bool) -> Vec<LlamaToken> {
    let c = CString::new(text).unwrap();
    let len = text.len() as c_int;
    // First call with empty buffer -> returns -(needed).
    let n = llama_tokenize(vocab, c.as_ptr(), len, std::ptr::null_mut(), 0, add_special, true);
    let need = (-n).max(0) as usize;
    if need == 0 {
        return Vec::new();
    }
    let mut toks = vec![0i32; need];
    let got = llama_tokenize(
        vocab,
        c.as_ptr(),
        len,
        toks.as_mut_ptr(),
        need as c_int,
        add_special,
        true,
    );
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
