{
  stdenv,
  cmake,
  fetchFromGitHub,
}:
stdenv.mkDerivation {
  name = "structopt";
  src = fetchFromGitHub {
    owner = "p-ranav";
    repo = "structopt";
    rev = "daac5f2";
    hash = "sha256-HSp1qSdlkOdFwrFCSpISdJz6a2MePsg8wE8xC9FMfQs";
  };
  nativeBuildInputs = [
    cmake
  ];
}
