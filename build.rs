// Link the in-process app against the prebuilt llama.cpp Vulkan runtime.
// libllama re-exports the ggml backend loader; the ggml backend .so's
// (incl. libggml-vulkan.so) are resolved at runtime via the rpath below.
use std::env;

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    // Single source of truth for the pinned build (also sourced by build.sh/clean.sh).
    let build = read_llama_build(&manifest);
    let libdir = format!("{manifest}/bin/llama-{build}");
    // Bake the default lib dir into the binaries; src/* uses env!("LLM_LIB_DIR_DEFAULT")
    // as the fallback when the LLM_LIB_DIR runtime override is unset.
    println!("cargo:rustc-env=LLM_LIB_DIR_DEFAULT={libdir}");

    println!("cargo:rustc-link-search=native={libdir}");
    println!("cargo:rustc-link-lib=dylib=llama");
    // ggml_backend_load_all (registers the Vulkan backend) is exported by libggml.
    println!("cargo:rustc-link-lib=dylib=ggml");

    // The linker is the default system `cc` (see .cargo/config.toml), so wrap
    // ld-style args in `-Wl,` so cc forwards them to the linker.
    // rpath: find libllama + the ggml backend libs at runtime without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-Wl,-rpath,{libdir}");
    // Transitive ggml symbols inside libllama resolve at load time via DT_NEEDED/rpath.
    println!("cargo:rustc-link-arg=-Wl,--allow-shlib-undefined");

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=llama-version.env");
}

// Parse LLAMA_BUILD out of llama-version.env (simple KEY=value lines).
fn read_llama_build(manifest: &str) -> String {
    let path = format!("{manifest}/llama-version.env");
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read {path}: {e}"));
    for line in contents.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("LLAMA_BUILD=") {
            return val.trim().trim_matches(['"', '\'']).to_string();
        }
    }
    panic!("LLAMA_BUILD not found in {path}");
}
