(function () {
  const dropzone = document.getElementById("dropzone");
  const viewerFrame = document.getElementById("viewer-frame");
  const fileInput = document.getElementById("file-input");
  const previewImg = document.getElementById("preview-img");
  const scanSweep = document.getElementById("scan-sweep");
  const analyzeBtn = document.getElementById("analyze-btn");
  const resetBtn = document.getElementById("reset-btn");
  const metaFilename = document.getElementById("meta-filename");
  const metaDims = document.getElementById("meta-dims");
  const errorBanner = document.getElementById("error-banner");

  const resultEmpty = document.getElementById("result-empty");
  const resultContent = document.getElementById("result-content");
  const verdictTag = document.getElementById("verdict-tag");
  const verdictClass = document.getElementById("verdict-class");
  const verdictConfidence = document.getElementById("verdict-confidence");
  const probBars = document.getElementById("prob-bars");
  const gradcamSwitch = document.getElementById("gradcam-switch");

  let currentFile = null;
  let gradcamOn = false;
  let lastResult = null;
  let originalImageSrc = null;

  function showError(message) {
    errorBanner.textContent = message;
    errorBanner.classList.add("show");
    setTimeout(() => errorBanner.classList.remove("show"), 5000);
  }

  function selectFile(file) {
    if (!file) return;
    currentFile = file;

    const url = URL.createObjectURL(file);
    originalImageSrc = url;
    previewImg.src = url;
    previewImg.style.display = "block";
    dropzone.style.display = "none";

    const img = new Image();
    img.onload = () => {
      metaDims.textContent = `${img.naturalWidth} x ${img.naturalHeight}px`;
    };
    img.src = url;

    metaFilename.textContent = file.name;
    analyzeBtn.disabled = false;

    resultContent.style.display = "none";
    resultEmpty.style.display = "block";
    lastResult = null;
  }

  dropzone.addEventListener("click", () => fileInput.click());
  fileInput.addEventListener("change", (e) => selectFile(e.target.files[0]));

  ["dragenter", "dragover"].forEach((evt) =>
    viewerFrame.addEventListener(evt, (e) => {
      e.preventDefault();
      viewerFrame.classList.add("drag-over");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    viewerFrame.addEventListener(evt, (e) => {
      e.preventDefault();
      viewerFrame.classList.remove("drag-over");
    })
  );
  viewerFrame.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    selectFile(file);
  });

  resetBtn.addEventListener("click", () => {
    currentFile = null;
    fileInput.value = "";
    previewImg.style.display = "none";
    dropzone.style.display = "block";
    metaFilename.textContent = "no file loaded";
    metaDims.textContent = "\u2014";
    analyzeBtn.disabled = true;
    resultContent.style.display = "none";
    resultEmpty.style.display = "block";
  });

  gradcamSwitch.addEventListener("click", () => {
    gradcamOn = !gradcamOn;
    gradcamSwitch.classList.toggle("on", gradcamOn);
    applyGradcamView();
  });

  function applyGradcamView() {
    if (!lastResult) return;
    if (gradcamOn && lastResult.gradcam_image) {
      previewImg.src = lastResult.gradcam_image;
    } else if (originalImageSrc) {
      previewImg.src = originalImageSrc;
    }
  }

  function renderResult(data) {
    lastResult = data;

    const tagClass = data.predicted_class;
    verdictTag.textContent = data.demo_mode ? `${tagClass} (simulated)` : tagClass;
    verdictTag.className = `tag ${tagClass}`;
    verdictClass.textContent = tagClass;
    verdictConfidence.textContent = `${data.confidence}%`;

    probBars.innerHTML = "";
    const entries = Object.entries(data.probabilities).sort((a, b) => b[1] - a[1]);
    entries.forEach(([label, value]) => {
      const row = document.createElement("div");
      row.className = "prob-row";
      row.innerHTML = `
        <span class="prob-label">${label}</span>
        <span class="prob-track"><span class="prob-fill ${label}" style="width:0%"></span></span>
        <span class="prob-value">${value}%</span>
      `;
      probBars.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector(".prob-fill").style.width = `${value}%`;
      });
    });

    if (!data.demo_mode && data.gradcam_image) {
      gradcamSwitch.style.pointerEvents = "auto";
      gradcamSwitch.style.opacity = "1";
    } else {
      gradcamOn = false;
      gradcamSwitch.classList.remove("on");
      gradcamSwitch.style.pointerEvents = "none";
      gradcamSwitch.style.opacity = "0.4";
    }
    applyGradcamView();

    resultEmpty.style.display = "none";
    resultContent.style.display = "block";
  }

  analyzeBtn.addEventListener("click", async () => {
    if (!currentFile) return;

    analyzeBtn.disabled = true;
    scanSweep.classList.add("active");

    const formData = new FormData();
    formData.append("scan", currentFile);

    try {
      const res = await fetch("/predict", { method: "POST", body: formData });
      const data = await res.json();

      if (!res.ok) {
        showError(data.error || "Prediction failed.");
      } else {
        renderResult(data);
      }
    } catch (err) {
      showError("Could not reach the server. Is app.py running?");
    } finally {
      scanSweep.classList.remove("active");
      analyzeBtn.disabled = false;
    }
  });
})();
