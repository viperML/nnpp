with import <nixpkgs> {};
mkShell.override {
  stdenv = llvmPackages.stdenv;
} {
  packages = [
    cmake
    neocmakelsp
    clang-tools
  ];

  env = {
    CMAKE_EXPORT_COMPILE_COMMANDS = true;
  };
}
