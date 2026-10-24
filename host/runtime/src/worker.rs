//! Engine worker process: discovery, spawn, log capture, kill.

use std::collections::VecDeque;
use std::io::{BufRead, BufReader};
use std::os::windows::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::Value;

use crate::rpc::RpcClient;

const CREATE_NO_WINDOW: u32 = 0x0800_0000;
const LOG_LINES: usize = 200;

#[derive(Debug, Clone)]
pub struct EngineLocation {
    pub dir: PathBuf,
    pub python: PathBuf,
}

/// Find `engine/` (with its uv venv): $RESONA_ENGINE_DIR, then walking up from the executable
/// and the current directory.
pub fn locate_engine() -> Option<EngineLocation> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Ok(d) = std::env::var("RESONA_ENGINE_DIR") {
        candidates.push(PathBuf::from(d));
    }
    let mut roots: Vec<PathBuf> = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        roots.extend(exe.ancestors().skip(1).map(Path::to_path_buf));
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.extend(cwd.ancestors().map(Path::to_path_buf));
    }
    candidates.extend(roots.into_iter().map(|r| r.join("engine")));
    candidates.into_iter().find_map(|dir| {
        let python = dir.join(".venv").join("Scripts").join("python.exe");
        (dir.join("pyproject.toml").is_file() && python.is_file()).then_some(EngineLocation { dir, python })
    })
}

pub struct WorkerProcess {
    child: Mutex<Child>,
    pub rpc: Arc<RpcClient>,
    pub log: Arc<Mutex<VecDeque<String>>>,
}

impl WorkerProcess {
    pub fn spawn<N, C>(loc: &EngineLocation, shm_prefix: &str, stub: bool, on_notify: N, on_exit: C) -> std::io::Result<Self>
    where
        N: Fn(&str, Value) + Send + 'static,
        C: FnOnce() + Send + 'static,
    {
        let mut cmd = Command::new(&loc.python);
        cmd.args(["-u", "-m", "resona_engine.worker", "--shm", shm_prefix])
            .current_dir(&loc.dir)
            .env("PYTHONIOENCODING", "utf-8")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .creation_flags(CREATE_NO_WINDOW);
        if stub {
            cmd.arg("--stub");
        }
        let mut child = cmd.spawn()?;
        let stdin = child.stdin.take().expect("piped stdin");
        let stdout = child.stdout.take().expect("piped stdout");
        let stderr = child.stderr.take().expect("piped stderr");
        let log: Arc<Mutex<VecDeque<String>>> = Arc::default();
        let log2 = log.clone();
        std::thread::Builder::new()
            .name("resona-engine-log".into())
            .spawn(move || {
                for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                    log::info!(target: "engine", "{line}");
                    let mut l = log2.lock();
                    if l.len() >= LOG_LINES {
                        l.pop_front();
                    }
                    l.push_back(line);
                }
            })?;
        let rpc = RpcClient::new(stdin, stdout, on_notify, on_exit);
        Ok(Self { child: Mutex::new(child), rpc, log })
    }

    pub fn log_tail(&self, n: usize) -> Vec<String> {
        let l = self.log.lock();
        l.iter().skip(l.len().saturating_sub(n)).cloned().collect()
    }

    pub fn kill(&self) {
        let mut c = self.child.lock();
        let _ = c.kill();
        let _ = c.wait();
    }

    pub fn is_alive(&self) -> bool {
        matches!(self.child.lock().try_wait(), Ok(None))
    }
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        self.kill();
    }
}
