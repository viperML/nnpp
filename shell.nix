with import <nixpkgs> {};
mkShell {
  packages = [
    cmake
    neocmakelsp
    clang-tools
  ];

  env = {
    CMAKE_EXPORT_COMPILE_COMMANDS = true;
  };
}
