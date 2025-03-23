use std::env;
use std::fs;
use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    
    // Handle macOS specific linking
    if env::var("CARGO_CFG_TARGET_OS").unwrap_or_default() == "macos" {
        // Tell cargo to look for libraries in /usr/lib as well
        println!("cargo:rustc-link-search=native=/usr/lib");
        
        // Tell macOS to use relative paths for dyld libraries
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path/../lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib");
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/local/lib");
    }
    
    // Check if this is a release build being installed
    if env::var("PROFILE").unwrap() == "release" {
        if let Ok(install_dir) = env::var("CARGO_HOME") {
            let bin_dir = Path::new(&install_dir).join("bin");
            if bin_dir.exists() {
                println!("cargo:warning=Binary will be installed to: {}", bin_dir.display());
            }
        }
    }
    
    // Package Python module with release builds
    if env::var("PROFILE").unwrap() == "release" {
        if let Ok(out_dir) = env::var("OUT_DIR") {
            println!("cargo:warning=Building Python package in release mode");
        }
    }
}
