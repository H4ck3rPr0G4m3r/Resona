//! JSON-RPC 2.0 client over the worker's stdio (newline-delimited).

use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use crossbeam_channel::{bounded, Sender};
use parking_lot::Mutex;
use serde_json::{json, Value};

#[derive(Debug, Clone, thiserror::Error)]
pub enum RpcError {
    #[error("{message}")]
    Remote { code: i64, message: String },
    #[error("engine did not answer '{0}' in time")]
    Timeout(String),
    #[error("engine process exited")]
    Closed,
    #[error("rpc i/o: {0}")]
    Io(String),
}

type Pending = Arc<Mutex<HashMap<u64, Sender<Result<Value, RpcError>>>>>;

pub struct RpcClient {
    stdin: Mutex<ChildStdin>,
    pending: Pending,
    next: AtomicU64,
}

impl RpcClient {
    /// Start the reader thread. `on_notify(method, params)` receives notifications; `on_close`
    /// runs once when stdout reaches EOF (process exit).
    pub fn new<N, C>(stdin: ChildStdin, stdout: ChildStdout, on_notify: N, on_close: C) -> Arc<Self>
    where
        N: Fn(&str, Value) + Send + 'static,
        C: FnOnce() + Send + 'static,
    {
        let pending: Pending = Arc::default();
        let p2 = pending.clone();
        std::thread::Builder::new()
            .name("resona-rpc-reader".into())
            .spawn(move || {
                let reader = BufReader::new(stdout);
                for line in reader.lines() {
                    let Ok(line) = line else { break };
                    let Ok(msg) = serde_json::from_str::<Value>(&line) else {
                        log::warn!("engine sent non-JSON line: {line}");
                        continue;
                    };
                    if let Some(id) = msg.get("id").and_then(Value::as_u64) {
                        if let Some(tx) = p2.lock().remove(&id) {
                            let res = match msg.get("error") {
                                Some(err) => Err(RpcError::Remote {
                                    code: err.get("code").and_then(Value::as_i64).unwrap_or(-1),
                                    message: err.get("message").and_then(Value::as_str).unwrap_or("error").to_string(),
                                }),
                                None => Ok(msg.get("result").cloned().unwrap_or(Value::Null)),
                            };
                            let _ = tx.send(res);
                        }
                    } else if let Some(method) = msg.get("method").and_then(Value::as_str) {
                        on_notify(method, msg.get("params").cloned().unwrap_or(Value::Null));
                    }
                }
                for (_, tx) in p2.lock().drain() {
                    let _ = tx.send(Err(RpcError::Closed));
                }
                on_close();
            })
            .expect("spawn rpc reader");
        Arc::new(Self { stdin: Mutex::new(stdin), pending, next: AtomicU64::new(1) })
    }

    pub fn call(&self, method: &str, params: Value, timeout: Duration) -> Result<Value, RpcError> {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let (tx, rx) = bounded(1);
        self.pending.lock().insert(id, tx);
        let line = json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}).to_string();
        {
            let mut w = self.stdin.lock();
            if let Err(e) = w.write_all(line.as_bytes()).and_then(|_| w.write_all(b"\n")).and_then(|_| w.flush()) {
                self.pending.lock().remove(&id);
                return Err(RpcError::Io(e.to_string()));
            }
        }
        match rx.recv_timeout(timeout) {
            Ok(r) => r,
            Err(_) => {
                self.pending.lock().remove(&id);
                Err(RpcError::Timeout(method.to_string()))
            }
        }
    }
}
