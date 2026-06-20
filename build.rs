// Link the in-process app against the prebuilt llama.cpp Vulkan runtime.
// libllama re-exports the ggml backend loader; the ggml backend .so's
// (incl. libggml-vulkan.so) are resolved at runtime via the rpath below.
use std::env;

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").unwrap();
    let libdir = format!("{manifest}/bin/llama-b9670");

    println!("cargo:rustc-link-search=native={libdir}");
    println!("cargo:rustc-link-lib=dylib=llama");
    // ggml_backend_load_all (registers the Vulkan backend) is exported by libggml.
    println!("cargo:rustc-link-lib=dylib=ggml");

    // linker-flavor is `ld` (see .cargo/config.toml), so pass ld-style args.
    // rpath: find libllama + the ggml backend libs at runtime without LD_LIBRARY_PATH.
    println!("cargo:rustc-link-arg=-rpath={libdir}");
    // Transitive ggml symbols inside libllama resolve at load time via DT_NEEDED/rpath.
    println!("cargo:rustc-link-arg=--allow-shlib-undefined");

    println!("cargo:rerun-if-changed=build.rs");
}
