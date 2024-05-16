use glob::glob;
use std::path::Path;
use std::path::PathBuf;
use std::fs;
use std::process::Command;

fn main() {
    let build_dir = "./src/cuda/build/";
    fs::create_dir(&build_dir);

    let status = Command::new("sh")
        .arg("-c")
        .arg("cmake .. && make -j12")
        .current_dir(build_dir)
        .status()
        .expect("Failed to execute command");

    let cuda_build_dir_relative = "./src/cuda/build/core";
    let current_dir = std::env::current_dir().expect("Failed to get current directory");
    let cuda_build_dir = current_dir.join(cuda_build_dir_relative);

    let cuda_home = match std::env::var("CUDA_HOME") {
        Ok(val) => val,
        Err(_) => {
            match std::env::var("CUDA_PATH") {
                Ok(val) => val,
                Err(_) => {
                    eprintln!("CUDA_HOME or CUDA_PATH environment variable not set.");
                    std::process::exit(1);
                }
            }
        }
    };

    let cuda_lib_dir = format!("{}/lib64", cuda_home);
    let cuda_include_dir = format!("{}/include", cuda_home);

    println!("CUDA library directory: {}", cuda_lib_dir);
    println!("CUDA include directory: {}", cuda_include_dir);

    println!(
        "cargo:rustc-link-search=native={}",
        cuda_build_dir.display()
    );
    println!("cargo:rustc-link-lib=static=panda-cuda");

    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=dylib=curand");

    println!("cargo:rustc-link-search=native={}", cuda_lib_dir);
    println!("cargo:rustc-link-lib=dylib=cudart");

    println!("cargo:rerun-if-env-changed=LD_LIBRARY_PATH");
    println!("cargo:rustc-env=LD_LIBRARY_PATH={}", cuda_lib_dir);
    println!("cargo:rustc-link-search=native={}", cuda_include_dir);
    println!("cargo:rustc-link-lib=dylib=cuda");

    println!("cargo:rustc-link-lib=dylib=stdc++");
}
