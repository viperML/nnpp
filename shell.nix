with import <nixpkgs> { };
mkShell.override { stdenv = clangStdenv; } {
  packages = [
    cmake
    ninja
    pkg-config
    neocmakelsp
    clang-tools
    unzip
    gtest
  ];

  hardeningDisable = ["all"];

  env = {
    CMAKE_EXPORT_COMPILE_COMMANDS = true;
    CMAKE_GENERATOR = "Ninja";
  };
}
