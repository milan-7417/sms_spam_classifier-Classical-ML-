/**
 * SpamGuard — SMS Spam Classifier Frontend Application
 * Handles asynchronous classification requests, state transitions,
 * input validation, and interactive result displays.
 */

document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements
  const form = document.getElementById("spam-form");
  const textarea = document.getElementById("message-input");
  const analyzeBtn = document.getElementById("analyze-btn");
  const clearBtn = document.getElementById("clear-btn");
  const charCounter = document.getElementById("char-counter");

  // States & Cards
  const loadingState = document.getElementById("loading-state");
  const errorCard = document.getElementById("error-card");
  const errorMessage = document.getElementById("error-message");
  const resultCard = document.getElementById("result-card");

  // Result Elements
  const resultIcon = document.getElementById("result-icon");
  const resultStatusText = document.getElementById("result-status-text");
  const riskBadge = document.getElementById("risk-badge");
  const explanationText = document.getElementById("explanation-text");

  // Sample Buttons
  const samplePills = document.querySelectorAll(".pill-btn");

  const MAX_CHARS = 1600;

  /**
   * Update character counter and button disabled state
   */
  function updateInputState() {
    if (!textarea || !charCounter || !analyzeBtn) return;
    const text = textarea.value;
    const len = text.length;
    const trimmed = text.trim();

    charCounter.textContent = `${len} / ${MAX_CHARS}`;

    if (len > MAX_CHARS - 100) {
      charCounter.classList.add("warning");
    } else {
      charCounter.classList.remove("warning");
    }

    if (trimmed.length > 0 && len <= MAX_CHARS) {
      analyzeBtn.removeAttribute("disabled");
    } else {
      analyzeBtn.setAttribute("disabled", "true");
    }
  }

  /**
   * Reset form and clear results
   */
  function handleClear() {
    if (!textarea) return;
    textarea.value = "";
    updateInputState();
    hideResults();
    hideError();
    textarea.focus();
  }

  function hideResults() {
    if (!resultCard) return;
    resultCard.classList.add("hidden");
    resultCard.classList.remove("is-spam", "is-ham");
  }

  function hideError() {
    if (!errorCard || !errorMessage) return;
    errorCard.classList.add("hidden");
    errorMessage.textContent = "";
  }

  function showError(msg) {
    hideResults();
    if (loadingState) loadingState.classList.add("hidden");
    if (errorMessage) errorMessage.textContent = msg;
    if (errorCard) errorCard.classList.remove("hidden");
  }

  /**
   * Render dynamic prediction results
   */
  function renderPredictionResult(data) {
    if (loadingState) loadingState.classList.add("hidden");
    hideError();

    const isSpam = data.prediction === "spam";

    if (resultCard) {
      // Card state styles
      resultCard.classList.remove("is-spam", "is-ham");
      resultCard.classList.add(isSpam ? "is-spam" : "is-ham");
    }

    // Header badge
    if (resultIcon) {
      resultIcon.textContent = isSpam ? "🚨" : "✓";
    }
    if (resultStatusText) {
      resultStatusText.textContent = isSpam ? "SPAM DETECTED" : "LEGITIMATE MESSAGE";
    }

    // Risk badge
    if (riskBadge) {
      riskBadge.textContent = isSpam ? "High Risk" : "Low Risk";
    }

    // Explanation
    if (explanationText) {
      explanationText.textContent = data.explanation || "";
    }

    // Show result card
    if (resultCard) {
      resultCard.classList.remove("hidden");
      resultCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  /**
   * Submit SMS message for analysis
   */
  async function handleSubmit(e) {
    if (e) e.preventDefault();
    if (!textarea) return;

    const rawMessage = textarea.value;
    const trimmed = rawMessage.trim();

    if (!trimmed) {
      showError("Please enter an SMS message to classify.");
      return;
    }

    if (rawMessage.length > MAX_CHARS) {
      showError(`Message length cannot exceed ${MAX_CHARS} characters.`);
      return;
    }

    // Enter loading state
    hideError();
    hideResults();
    if (loadingState) loadingState.classList.remove("hidden");
    if (analyzeBtn) analyzeBtn.setAttribute("disabled", "true");

    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        body: JSON.stringify({ message: trimmed }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || data.error || `Server returned error (${response.status})`);
      }

      renderPredictionResult(data);
    } catch (err) {
      showError(err.message || "Failed to analyze message. Please check the server connection.");
    } finally {
      if (loadingState) loadingState.classList.add("hidden");
      updateInputState();
    }
  }

  // Event Listeners
  if (textarea) {
    textarea.addEventListener("input", updateInputState);
    textarea.addEventListener("change", updateInputState);
  }
  if (form) {
    form.addEventListener("submit", handleSubmit);
  }
  if (clearBtn) {
    clearBtn.addEventListener("click", handleClear);
  }

  // Quick Sample Buttons
  samplePills.forEach((pill) => {
    pill.addEventListener("click", () => {
      const sampleText = pill.getAttribute("data-sample");
      if (sampleText && textarea) {
        textarea.value = sampleText;
        updateInputState();
        handleSubmit();
      }
    });
  });

  // Initial setup
  updateInputState();
});
