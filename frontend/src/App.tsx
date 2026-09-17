import { useEffect, useState } from "react";
import { fetchHealth } from "./api/client";

type BackendStatus = "checking" | "connected" | "unreachable";

function App() {
  const [backendStatus, setBackendStatus] = useState<BackendStatus>("checking");

  useEffect(() => {
    fetchHealth()
      .then(() => setBackendStatus("connected"))
      .catch(() => setBackendStatus("unreachable"));
  }, []);

  return (
    <main className="page">
      <h1>Keyword Intelligence</h1>
      <p className="subtitle">
        Local keyword extraction and ranking pipeline. The real analysis UI
        is built in later phases — this is the Phase 1 scaffold.
      </p>

      <p className="status">
        Backend status:{" "}
        <span className={`status-badge status-${backendStatus}`}>
          {backendStatus}
        </span>
      </p>
    </main>
  );
}

export default App;
