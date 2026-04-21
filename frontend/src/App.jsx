import React, { useEffect, useState } from "react";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState("");
  const [lenslessFile, setLenslessFile] = useState(null);
  const [lensedFile, setLensedFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  useEffect(() => {
    async function fetchModels() {
      try {
        const res = await fetch(`${API_BASE}/models`);
        const data = await res.json();
        setModels(data.models || []);
        if (data.models?.length) {
          setSelectedModel(data.models[0].name);
        }
      } catch (err) {
        console.error(err);
        setError("Failed to load models.");
      }
    }

    fetchModels();
  }, []);

  async function handleSubmit(e) {
    e.preventDefault();
    setError("");
    setResult(null);

    if (!lenslessFile) {
      setError("Please choose a lensless input file.");
      return;
    }

    if (!selectedModel) {
      setError("Please select a model.");
      return;
    }

    const formData = new FormData();
    formData.append("lensless_file", lenslessFile);
    formData.append("model_name", selectedModel);

    if (lensedFile) {
      formData.append("lensed_file", lensedFile);
    }

    try {
      setLoading(true);

      const res = await fetch(`${API_BASE}/reconstruct`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || "Reconstruction failed.");
      }

      setResult(data);
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="page">
      <div className="card">
        <h1>Lensless Reconstruction</h1>
        <p className="subtext">
          Upload a lensless .npy file, optionally add a lensed reference image,
          and compare reconstructions.
        </p>

        <form onSubmit={handleSubmit} className="form">
          <label>
            Model
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.display_name}
                </option>
              ))}
            </select>
          </label>

          <label>
            Lensless file
            <input
              type="file"
              accept=".npy,.png,.jpg,.jpeg,.bmp,.tif,.tiff"
              onChange={(e) => setLenslessFile(e.target.files?.[0] || null)}
            />
          </label>

          <label>
            Optional lensed file
            <input
              type="file"
              accept=".npy,.png,.jpg,.jpeg,.bmp,.tif,.tiff"
              onChange={(e) => setLensedFile(e.target.files?.[0] || null)}
            />
          </label>

          <button type="submit" disabled={loading}>
            {loading ? "Running..." : "Run reconstruction"}
          </button>
        </form>

        {error && <div className="error">{error}</div>}

        {result && (
          <div className="result">
            <h2>Inputs and outputs</h2>
            <div className="image-grid">
              <div className="image-card">
                <h3>Lensless input</h3>
                <img
                  src={`${API_BASE}${result.lensless_preview_url}`}
                  alt="Lensless preview"
                  className="output-image"
                />
              </div>

              {result.lensed_preview_url && (
                <div className="image-card">
                  <h3>Lensed reference</h3>
                  <img
                    src={`${API_BASE}${result.lensed_preview_url}`}
                    alt="Lensed preview"
                    className="output-image"
                  />
                </div>
              )}

              {result.mode === "single" && (
                <div className="image-card">
                  <h3>{result.result.display_name}</h3>
                  <img
                    src={`${API_BASE}${result.result.image_url}`}
                    alt={result.result.display_name}
                    className="output-image"
                  />
                  <div className="meta">
                    <div>
                      <strong>Inference time:</strong>{" "}
                      {result.result.inference_time_ms} ms
                    </div>
                    <div>
                      <strong>Input shape:</strong>{" "}
                      {JSON.stringify(result.input_shape)}
                    </div>
                  </div>
                </div>
              )}

              {result.mode === "all" &&
                result.results.map((item) => (
                  <div className="image-card" key={item.name}>
                    <h3>{item.display_name}</h3>
                    <img
                      src={`${API_BASE}${item.image_url}`}
                      alt={item.display_name}
                      className="output-image"
                    />
                    <div className="meta">
                      <div>
                        <strong>Inference time:</strong>{" "}
                        {item.inference_time_ms} ms
                      </div>
                    </div>
                  </div>
                ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}