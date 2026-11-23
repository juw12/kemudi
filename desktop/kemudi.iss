; kemudi for Windows - per-user installer (no admin rights). build.py compiles it with:
;   iscc /DAppVersion=1.0.0 /DSourceDir=<dist\kemudi> /DOutputDir=<dist> /DIconFile=<build\kemudi.ico> kemudi.iss
; Installs to %LOCALAPPDATA%\Programs\kemudi, adds kemudi to the Start menu (and optionally the desktop) and lists it
; in Settings > Apps. The window keeps its data (what you saved in it) in %LOCALAPPDATA%\kemudi, which
; uninstalling leaves in place so a reinstall finds it again.

#ifndef AppVersion
  #define AppVersion "1.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "dist\kemudi"
#endif
#ifndef OutputDir
  #define OutputDir "dist"
#endif
#ifndef IconFile
  #define IconFile "build\kemudi.ico"
#endif

[Setup]
AppId={{E81247BA-698B-569E-A6DE-E1E1C366EEFB}
AppName=kemudi
AppVersion={#AppVersion}
AppVerName=kemudi {#AppVersion}
AppPublisher=Juwen Lie
AppPublisherURL=https://juw12.github.io/kemudi/
AppSupportURL=https://github.com/juw12/kemudi
DefaultDirName={autopf}\kemudi
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir={#OutputDir}
OutputBaseFilename=kemudi-Windows-Setup
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\kemudi.exe
UninstallDisplayName=kemudi
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
CloseApplications=no
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{userprograms}\kemudi"; Filename: "{app}\kemudi.exe"; Comment: "Steering knobs for a local language model"
Name: "{userdesktop}\kemudi"; Filename: "{app}\kemudi.exe"; Comment: "Steering knobs for a local language model"; Tasks: desktopicon

[Run]
Filename: "{app}\kemudi.exe"; Description: "{cm:LaunchProgram,kemudi}"; Flags: nowait postinstall skipifsilent

[UninstallRun]
Filename: "{app}\kemudi.exe"; Parameters: "--stop"; Flags: runhidden waituntilterminated; RunOnceId: "Stopkemudi"

[Code]
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ResultCode: Integer;
begin
  Result := '';
  { Updating: close an open kemudi window first so its files can be replaced. }
  if FileExists(ExpandConstant('{app}\kemudi.exe')) then
    Exec(ExpandConstant('{app}\kemudi.exe'), '--stop', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
end;
