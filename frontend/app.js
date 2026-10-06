/**
 * ELDERLY AI MONITOR — FRONTEND DASHBOARD CONTROLLER
 * Real-time WebSocket connection, REST telemetry polling, and MJPEG video streaming.
 * Authentication layer with JWT token management.
 */

(() => {
  "use strict";

  // 1. API Configuration, SSL & Auto-Discovery
  const isHttps = window.location.protocol === "https:";
  const isFileOrigin = window.location.protocol === "file:";
  const API_BASE = isFileOrigin ? "http://127.0.0.1:8000" : window.location.origin;
  const WS_PROTOCOL = isHttps ? "wss:" : "ws:";
  const WS_BASE = API_BASE.replace(/^https?:/, WS_PROTOCOL);

  const AUTH_STORAGE_KEY = "fall_stroke_auth_token";
  const REFRESH_STORAGE_KEY = "fall_stroke_refresh_token";

  function getSavedToken() {
    try {
      return localStorage.getItem(AUTH_STORAGE_KEY) || "";
    } catch (_) {
      return "";
    }
  }

  function saveAuthToken(token) {
    try {
      localStorage.setItem(AUTH_STORAGE_KEY, token);
    } catch (_) {}
  }

  function clearAuthToken() {
    try {
      localStorage.removeItem(AUTH_STORAGE_KEY);
      localStorage.removeItem(REFRESH_STORAGE_KEY);
    } catch (_) {}
  }

  function saveRefreshToken(token) {
    try {
      if (token) localStorage.setItem(REFRESH_STORAGE_KEY, token);
    } catch (_) {}
  }

  function getRefreshToken() {
    try {
      return localStorage.getItem(REFRESH_STORAGE_KEY) || "";
    } catch (_) {
      return "";
    }
  }

  // Access tokens are short-lived (security.yaml): renew silently with the refresh token.
  // Concurrent 401s share one refresh request.
  let refreshInFlight = null;
  function refreshAccessToken() {
    const refreshToken = getRefreshToken();
    if (!refreshToken) return Promise.resolve(false);
    if (!refreshInFlight) {
      refreshInFlight = fetch(`${API_BASE}/api/auth/refresh`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ refresh_token: refreshToken }),
      })
        .then(async (res) => {
          if (!res.ok) {
            if (res.status === 401) clearAuthToken();
            return false;
          }
          const json = await res.json();
          saveAuthToken(json.access_token);
          return true;
        })
        .catch(() => false)
        .finally(() => { refreshInFlight = null; });
    }
    return refreshInFlight;
  }

  // Helper for MJPEG stream URL with cache-buster and token parameter
  function buildMjpegUrl() {
    const token = getSavedToken();
    const tokenParam = token ? `&token=${encodeURIComponent(token)}` : "";
    return `${API_BASE}/stream/mjpeg?t=${Date.now()}${tokenParam}`;
  }

  function getWsUrl() {
    const token = getSavedToken();
    return `${WS_BASE}/ws/alerts${token ? `?token=${encodeURIComponent(token)}` : ""}`;
  }

  // Wrapper around native fetch that automatically injects JWT Bearer token
  async function apiFetch(url, options = {}) {
    const opts = { ...options };
    const headers = new Headers(opts.headers || {});
    const token = getSavedToken();
    if (token && !headers.has("Authorization")) {
      headers.set("Authorization", `Bearer ${token}`);
    }
    opts.headers = headers;

    let res = await fetch(url, opts);
    const isAuthCall = url.includes("/api/auth/");
    if (res.status === 401 && !isAuthCall && token && await refreshAccessToken()) {
      headers.set("Authorization", `Bearer ${getSavedToken()}`);
      res = await fetch(url, { ...opts, headers });
    }
    if (res.status === 401 && !isAuthCall) {
      handleAuthRequired();
    }
    return res;
  }

  // State
  let soundEnabled = true;
  let isCameraRunning = false;
  let alertCount = 0;
  let wsConnection = null;
  let wsReconnectTimer = null;

  // DOM Elements
  const elBackendDot = document.getElementById("backend-status-dot");
  const elBackendText = document.getElementById("backend-status-text");
  const elWsDot = document.getElementById("ws-status-dot");
  const elWsText = document.getElementById("ws-status-text");
  const elClock = document.getElementById("digital-clock");

  const elKpiFps = document.getElementById("kpi-fps");
  const elKpiTracks = document.getElementById("kpi-tracks");
  const elKpiRisk = document.getElementById("kpi-risk");
  const elKpiRiskState = document.getElementById("kpi-risk-state");
  const elKpiRiskMeter = document.getElementById("kpi-risk-meter");

  const elTelemCpu = document.getElementById("telem-cpu");
  const elTelemRam = document.getElementById("telem-ram");
  const elTelemGpu = document.getElementById("telem-gpu");

  const elVideoStream = document.getElementById("live-video-stream");
  const elOfflineOverlay = document.getElementById("stream-offline-overlay");

  // Camera Management DOM Elements
  const elCameraTypeSelect = document.getElementById("camera-type-select");
  const elGroupWebcamDevice = document.getElementById("group-webcam-device");
  const elWebcamDeviceSelect = document.getElementById("webcam-device-select");
  const elBtnScanWebcams = document.getElementById("btn-scan-webcams");

  const elGroupRtspPreset = document.getElementById("group-rtsp-preset");
  const elRtspPresetSelect = document.getElementById("rtsp-preset-select");
  const elGroupRtspUrlRow = document.getElementById("group-rtsp-url-row");
  const elRtspUrlInput = document.getElementById("rtsp-url-input");
  const elRtspUsernameInput = document.getElementById("rtsp-username-input");
  const elRtspPasswordInput = document.getElementById("rtsp-password-input");
  const elBtnTestRtsp = document.getElementById("btn-test-rtsp");
  const elBtnDiscoverCameras = document.getElementById("btn-discover-cameras");
  const elCamDiscoverResults = document.getElementById("cam-discover-results");

  const elGroupVideoPreset = document.getElementById("group-video-preset");
  const elVideoPresetSelect = document.getElementById("video-preset-select");
  const elGroupVideoCustomRow = document.getElementById("group-video-custom-row");
  const elCustomVideoPathInput = document.getElementById("custom-video-path-input");

  const elBtnApplyCamera = document.getElementById("btn-apply-camera");
  const elBtnApplyTxt = document.getElementById("btn-apply-txt");
  const elCamSpinner = document.getElementById("cam-spinner");
  const elBtnSaveCameraDefault = document.getElementById("btn-save-camera-default");
  const elBtnCameraToggle = document.getElementById("btn-camera-toggle");
  const elBtnCamToggleIcon = document.getElementById("btn-cam-toggle-icon");
  const elBtnCamToggleTxt = document.getElementById("btn-cam-toggle-txt");
  const elBtnRefreshStream = document.getElementById("btn-refresh-stream");

  const elCamToastFeedback = document.getElementById("cam-toast-feedback");
  const elCamActiveStatusPill = document.getElementById("cam-active-status-pill");
  const elCamActiveNameTxt = document.getElementById("cam-active-name-txt");
  const elHudCamName = document.getElementById("hud-cam-name");

  const elEmergencyBar = document.getElementById("emergency-alert-bar");
  const elBtnDismissAlert = document.getElementById("btn-dismiss-alert");
  const elAlertBarDesc = document.getElementById("alert-bar-desc");

  const elAlertsFeed = document.getElementById("alerts-feed");
  const elEmptyPlaceholder = document.getElementById("empty-alerts-placeholder");
  const elAlertsCountBadge = document.getElementById("alerts-count-badge");
  const elBtnTestChime = document.getElementById("btn-test-chime");
  const elBtnClearTemp = document.getElementById("btn-clear-temp");
  const elBtnClearPermanent = document.getElementById("btn-clear-permanent");
  const elBtnSoundToggle = document.getElementById("btn-sound-toggle");
  const elBtnFullscreen = document.getElementById("btn-fullscreen");
  const elPersonTableBody = document.getElementById("person-table-body");

  // Video container & Fall Visual Elements
  const elVideoContainer = document.getElementById("video-container");
  const elVideoCard = document.querySelector(".video-card");
  const elVideoFallBadge = document.getElementById("video-fall-badge");
  const elVideoFallText = document.getElementById("video-fall-text");

  // Authentication Modal DOM Elements
  const elAuthModalBackdrop = document.getElementById("auth-modal-backdrop");
  const elAuthModalTitle = document.getElementById("auth-modal-title");
  const elAuthModalSubtitle = document.getElementById("auth-modal-subtitle");
  const elAuthForm = document.getElementById("auth-form");
  const elAuthUsername = document.getElementById("auth-username-input");
  const elAuthPassword = document.getElementById("auth-password-input");
  const elAuthError = document.getElementById("auth-error-msg");
  const elBtnAuthSubmit = document.getElementById("btn-auth-submit");
  const elAuthSubmitTxt = document.getElementById("auth-submit-txt");

  let isSetupMode = false;
  let isDashboardInitialized = false;

  // Web Audio API Synthesizer for Multi-harmonic Emergency Alarm Chimes & Horn
  let audioCtx = null;
  let alarmChimeInterval = null;
  let alarmChimeTimeout = null;
  let sirenInterval = null;
  let sirenTimeout = null;
  let borderFlashTimeout = null;

  function initAudioContext() {
    if (!audioCtx) {
      const AudioContextClass = window.AudioContext || window.webkitAudioContext;
      if (AudioContextClass) {
        audioCtx = new AudioContextClass();
      }
    }
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume().catch(() => {});
    }
  }

  // Tự động mở khóa AudioContext khi người dùng chạm hoặc click bất kỳ đâu
  ["click", "keydown", "touchstart", "pointerdown"].forEach((evt) => {
    window.addEventListener(evt, () => initAudioContext(), { passive: true });
  });

  // Tạo âm chuông kim loại vang đa hòa âm (Metallic Chime Bell Harmonic Synthesis)
  function playBellChimeTone(freq, duration = 0.55, gainLevel = 0.5) {
    if (!soundEnabled) return;
    initAudioContext();
    if (!audioCtx) return;

    try {
      const now = audioCtx.currentTime;

      // 1. Tần số cơ bản (Fundamental Sine Tone)
      const osc1 = audioCtx.createOscillator();
      const gain1 = audioCtx.createGain();
      osc1.type = "sine";
      osc1.frequency.setValueAtTime(freq, now);
      gain1.gain.setValueAtTime(gainLevel, now);
      gain1.gain.exponentialRampToValueAtTime(0.001, now + duration);
      osc1.connect(gain1);
      gain1.connect(audioCtx.destination);
      osc1.start(now);
      osc1.stop(now + duration);

      // 2. Họa âm quãng 8 (Octave Overtone - tạo âm vang chuông thật)
      const osc2 = audioCtx.createOscillator();
      const gain2 = audioCtx.createGain();
      osc2.type = "sine";
      osc2.frequency.setValueAtTime(freq * 2.015, now);
      gain2.gain.setValueAtTime(gainLevel * 0.45, now);
      gain2.gain.exponentialRampToValueAtTime(0.001, now + duration * 0.75);
      osc2.connect(gain2);
      gain2.connect(audioCtx.destination);
      osc2.start(now);
      osc2.stop(now + duration * 0.75);

      // 3. Họa âm ánh kim bậc cao (Triangle Shimmer)
      const osc3 = audioCtx.createOscillator();
      const gain3 = audioCtx.createGain();
      osc3.type = "triangle";
      osc3.frequency.setValueAtTime(freq * 3.01, now);
      gain3.gain.setValueAtTime(gainLevel * 0.25, now);
      gain3.gain.exponentialRampToValueAtTime(0.001, now + duration * 0.45);
      osc3.connect(gain3);
      gain3.connect(audioCtx.destination);
      osc3.start(now);
      osc3.stop(now + duration * 0.45);

    } catch (e) {
      console.warn("[Dashboard] Bell chime error:", e);
    }
  }

  // Phát tiếng còi báo động khẩn cấp
  function playSirenBeep(freq, duration = 0.22) {
    if (!soundEnabled) return;
    initAudioContext();
    if (!audioCtx) return;
    try {
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();

      osc.type = "sawtooth"; // Tiếng còi cứu thương khẩn cấp
      osc.frequency.setValueAtTime(freq, audioCtx.currentTime);

      gain.gain.setValueAtTime(0.35, audioCtx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + duration);

      osc.connect(gain);
      gain.connect(audioCtx.destination);

      osc.start();
      osc.stop(audioCtx.currentTime + duration);
    } catch (e) {
      console.warn("[Dashboard] Audio error:", e);
    }
  }

  // 1. Chuỗi chuông báo động ngã khẩn cấp (Fall Risk Urgent Alarm Chime: C6 -> A5 -> F6 -> C6 + horn)
  function playFallAlarmChimeSequence() {
    const notes = [1046.5, 880.0, 1396.9, 1046.5];
    notes.forEach((freq, idx) => {
      setTimeout(() => {
        if (!soundEnabled) return;
        playBellChimeTone(freq, 0.45, 0.5);
      }, idx * 140);
    });

    setTimeout(() => {
      if (!soundEnabled) return;
      playSirenBeep(960, 0.25);
    }, 600);
  }

  // 2. Chuỗi chuông báo hành vi bất thường (Abnormal Behavior Warning Chime: Đinh - Đoong)
  function playAbnormalWarningChimeSequence() {
    playBellChimeTone(880.0, 0.6, 0.45);
    setTimeout(() => {
      if (!soundEnabled) return;
      playBellChimeTone(659.25, 0.8, 0.5);
    }, 240);
  }

  // Bộ phát âm thanh chuông cảnh báo ra loa
  function triggerAlarmChime(type = "fall") {
    if (!soundEnabled) return;
    initAudioContext();
    stopContinuousAlarm();

    if (type === "fall") {
      // Chuông báo té ngã: Kêu dồn dập mỗi 1.4 giây
      playFallAlarmChimeSequence();
      alarmChimeInterval = setInterval(() => {
        if (!soundEnabled) {
          stopContinuousAlarm();
          return;
        }
        playFallAlarmChimeSequence();
      }, 1400);

      // Tự động dừng sau 10 giây nếu chưa có ai nhấn xác nhận
      alarmChimeTimeout = setTimeout(() => {
        stopContinuousAlarm();
      }, 10000);
    } else {
      // Chuông báo hành vi bất thường: Kêu nhắc nhở mỗi 2.2 giây
      playAbnormalWarningChimeSequence();
      alarmChimeInterval = setInterval(() => {
        if (!soundEnabled) {
          stopContinuousAlarm();
          return;
        }
        playAbnormalWarningChimeSequence();
      }, 2200);

      alarmChimeTimeout = setTimeout(() => {
        stopContinuousAlarm();
      }, 7000);
    }

    // Dự phòng âm thanh bằng thẻ HTML5 <audio>
    const elAudio = document.getElementById("alert-sound");
    if (elAudio) {
      try {
        elAudio.currentTime = 0;
        elAudio.play().catch(() => {});
      } catch (_) {}
    }
  }

  function stopContinuousAlarm() {
    if (alarmChimeInterval) {
      clearInterval(alarmChimeInterval);
      alarmChimeInterval = null;
    }
    if (alarmChimeTimeout) {
      clearTimeout(alarmChimeTimeout);
      alarmChimeTimeout = null;
    }
    if (sirenInterval) {
      clearInterval(sirenInterval);
      sirenInterval = null;
    }
    if (sirenTimeout) {
      clearTimeout(sirenTimeout);
      sirenTimeout = null;
    }
  }

  function clearEmergencyVisuals() {
    if (elVideoContainer) elVideoContainer.classList.remove("fall-emergency-border");
    if (elVideoCard) elVideoCard.classList.remove("fall-emergency-border");
    if (elVideoFallBadge) elVideoFallBadge.style.display = "none";
    if (borderFlashTimeout) {
      clearTimeout(borderFlashTimeout);
      borderFlashTimeout = null;
    }
  }

  // 2. Clock updater
  const elVideoOverlayTs = document.getElementById("video-overlay-ts");
  const pad2 = (n) => String(n).padStart(2, "0");
  function updateClock() {
    const now = new Date();
    elClock.textContent = now.toTimeString().split(" ")[0];
    if (elVideoOverlayTs) {
      // Local date-time on the live video, e.g. 2026-10-04 15:42:08
      elVideoOverlayTs.textContent =
        `${now.getFullYear()}-${pad2(now.getMonth() + 1)}-${pad2(now.getDate())} ` +
        `${pad2(now.getHours())}:${pad2(now.getMinutes())}:${pad2(now.getSeconds())}`;
    }
  }
  setInterval(updateClock, 1000);
  updateClock();

  // 3. WebSocket Alerts Listener
  function connectAlertsWebSocket() {
    if (wsConnection) {
      try { wsConnection.close(); } catch (_) {}
    }

    elWsDot.className = "badge-dot dot-warning";
    elWsText.textContent = "WebSocket: Đang nối...";

    try {
      wsConnection = new WebSocket(getWsUrl());

      wsConnection.onopen = () => {
        elWsDot.className = "badge-dot dot-online";
        elWsText.textContent = "WebSocket: Real-time";
        if (wsReconnectTimer) {
          clearTimeout(wsReconnectTimer);
          wsReconnectTimer = null;
        }
      };

      wsConnection.onmessage = (event) => {
        try {
          const alertData = JSON.parse(event.data);
          handleIncomingAlert(alertData);
        } catch (err) {
          console.error("[Dashboard] Error parsing WebSocket message:", err);
        }
      };

      wsConnection.onclose = (evt) => {
        elWsDot.className = "badge-dot dot-offline";
        elWsText.textContent = "WebSocket: Mất kết nối";
        if (evt && evt.code === 4001) {
          handleAuthRequired();
        } else {
          scheduleWsReconnect();
        }
      };

      wsConnection.onerror = () => {
        elWsDot.className = "badge-dot dot-offline";
        elWsText.textContent = "WebSocket: Lỗi";
      };
    } catch (e) {
      console.warn("[Dashboard] WebSocket init failed, will retry:", e);
      scheduleWsReconnect();
    }
  }

  function scheduleWsReconnect() {
    if (!wsReconnectTimer) {
      wsReconnectTimer = setTimeout(() => {
        wsReconnectTimer = null;
        connectAlertsWebSocket();
      }, 3000);
    }
  }

  // 4. Handle Incoming Alert
  function handleIncomingAlert(alert) {
    const personId = alert.person_id ?? "N/A";
    const primaryAction = (alert.evidence?.primary_action ?? alert.action ?? "").toLowerCase();
    const riskScore = typeof alert.risk_score === "number" ? alert.risk_score : 0.95;
    // Alerts without camera_id come from the primary camera (big video); others from an extra camera tile
    const fromExtraCamera = Boolean(alert.camera_id);
    const cameraLabel = alert.camera_name || "Camera chính";

    // Hai luồng riêng: "fall" (té ngã) và "stroke_warning" (dấu hiệu cảnh báo đột quỵ / bệnh cấp).
    // Sự kiện cũ (chưa có event_type) vẫn phân loại theo nhãn hành động như trước.
    const eventType = alert.event_type || "";
    const isStrokeSigns = eventType === "stroke_warning";
    const isFallRisk = eventType === "fall" || (!isStrokeSigns && (
      primaryAction === "falling" ||
      primaryAction === "stumbling" ||
      primaryAction === "lying" ||
      primaryAction === "immobile" ||
      riskScore >= 0.70 ||
      (alert.evidence?.drop_severity_score && alert.evidence.drop_severity_score > 0.35)
    ));
    const strokeSignsText = (alert.evidence?.stroke_signs || []).map((w) => LYING_CONTEXT_LABELS[w] || w).join(", ");

    // 1. Show flash emergency banner
    elEmergencyBar.style.display = "block";
    const alertTypeLabel = isStrokeSigns
      ? `DẤU HIỆU CẢNH BÁO ĐỘT QUỴ / BỆNH CẤP${strokeSignsText ? `: ${strokeSignsText}` : ""}`
      : (isFallRisk ? "NGUY CƠ TÉ NGÃ KHẨN CẤP" : "HÀNH VI BẤT THƯỜNG");
    elAlertBarDesc.textContent = `[${alertTypeLabel}] ${cameraLabel} — Đối tượng ID: #${personId} — ${primaryAction.toUpperCase()} (Điểm rủi ro: ${riskScore.toFixed(2)}). Possible medical emergency / abnormal behavior detected. Please check the person.`;

    // 2. Hiện khung viền màu đỏ chớp nháy cảnh báo & Floating badge trên video (hoặc trên ô camera thêm)
    if (fromExtraCamera) flashExtraCameraTile(alert.camera_id);
    if (!fromExtraCamera && elVideoContainer) elVideoContainer.classList.add("fall-emergency-border");
    if (!fromExtraCamera && elVideoCard) elVideoCard.classList.add("fall-emergency-border");
    if (!fromExtraCamera && elVideoFallBadge) {
      elVideoFallBadge.style.display = "flex";
      if (elVideoFallText) {
        elVideoFallText.textContent = isStrokeSigns
          ? `⚠️ DẤU HIỆU CẢNH BÁO ĐỘT QUỴ / BỆNH CẤP — HÃY KIỂM TRA NGƯỜI NÀY! (ID #${personId})`
          : isFallRisk
            ? `🚨 CẢNH BÁO: PHÁT HIỆN NGUY CƠ TÉ NGÃ! (ID #${personId})`
            : `⚠️ CẢNH BÁO: PHÁT HIỆN HÀNH VI BẤT THƯỜNG! (ID #${personId})`;
      }
    }

    // Tự động tắt viền đỏ sau 10 giây nếu không có cảnh báo mới
    if (borderFlashTimeout) clearTimeout(borderFlashTimeout);
    borderFlashTimeout = setTimeout(() => {
      clearEmergencyVisuals();
    }, 10000);

    // 3. KÍCH HOẠT TIẾNG CHUÔNG CẢNH BÁO RA LOA (Chuông té ngã khẩn cấp hoặc chuông hành vi bất thường)
    triggerAlarmChime(isFallRisk ? "fall" : "abnormal");

    // 4. Cập nhật thẻ CẢNH BÁO TÉ NGÃ trên giao diện mới
    const elIncTs = document.getElementById("incident-timestamp-txt");
    const elIncConf = document.getElementById("incident-confidence-txt");
    const elIncStatus = document.getElementById("incident-status-tag");

    if (elIncTs) elIncTs.textContent = `${new Date().toLocaleTimeString()} - ${new Date().toLocaleDateString('vi-VN')}`;
    if (elIncConf) elIncConf.textContent = `${(riskScore * 100).toFixed(0)}%`;
    if (elIncStatus) {
      elIncStatus.textContent = "Đang xử lý";
      elIncStatus.style.background = "#fef2f2";
      elIncStatus.style.borderColor = "#fecaca";
      elIncStatus.style.color = "#dc2626";
    }

    // 5. Thêm dòng vào Bảng Lịch Sử Cảnh Báo
    const elAlertsTbody = document.getElementById("alerts-table-body");
    if (elAlertsTbody) {
      const tr = document.createElement("tr");
      tr.className = "alert-row-highlight";
      tr.innerHTML = `
        <td><span class="id-tag">#${String(alertCount + 1).padStart(3, '0')}</span></td>
        <td>${new Date().toLocaleTimeString()}</td>
        <td><span class="badge-type-danger">${isStrokeSigns ? "Dấu hiệu đột quỵ" : (isFallRisk ? "Té ngã" : "Bất thường")}</span></td>
        <td>${escapeHtml(cameraLabel)}</td>
        <td><span class="badge-state-pending">Đang xử lý</span></td>
      `;
      elAlertsTbody.prepend(tr);
    }

    // 6. Prepend to feed
    renderAlertCard(alert, true);

    // 7. Update KPI
    updateRiskKpi(riskScore, isFallRisk ? "Cao" : "Nghi vấn");
    const elTotalAlerts = document.getElementById("stat-total-alerts-count");
    if (elTotalAlerts) elTotalAlerts.textContent = alertCount;
  }

  // Fall clips are recorded for a few seconds after the alert, then kept for 7 days at most
  async function toggleAlertClip(card, eventId, btn) {
    const row = btn.parentElement;
    const existing = row.querySelector(".alert-item-clip, .alert-item-clip-note");
    if (existing) {
      row.querySelectorAll(".alert-item-clip, .alert-item-clip-note").forEach((el) => el.remove());
      btn.textContent = "▶ Xem đoạn video té ngã";
      return;
    }
    const token = getSavedToken();
    const url = `${API_BASE}/events/${encodeURIComponent(eventId)}/clip${token ? `?token=${encodeURIComponent(token)}` : ""}`;
    btn.disabled = true;
    try {
      const res = await apiFetch(`${API_BASE}/events/${encodeURIComponent(eventId)}`);
      const info = res.ok ? await res.json() : null;
      if (!info || !info.clip_available) {
        const msg = info
          ? "Đoạn video đang được ghi (hoặc đã hết hạn lưu 7 ngày). Vui lòng thử lại sau vài giây."
          : "Không tìm thấy sự kiện này trên máy chủ.";
        const note = document.createElement("div");
        note.className = "alert-item-clip-note";
        note.textContent = msg;
        row.appendChild(note);
        return;
      }
      const video = document.createElement("video");
      video.className = "alert-item-clip";
      video.src = url;
      video.controls = true;
      video.autoplay = true;
      video.muted = true;
      video.playsInline = true;
      row.appendChild(video);
      const note = document.createElement("div");
      note.className = "alert-item-clip-note";
      note.textContent = "Khuôn mặt đã được làm mờ. Video tự động xóa sau 7 ngày.";
      row.appendChild(note);
      btn.textContent = "✕ Đóng video";
    } catch (e) {
      console.warn("[Dashboard] Không tải được đoạn video:", e);
    } finally {
      btn.disabled = false;
    }
  }

  function renderAlertCard(alert, isNew = false) {
    if (elEmptyPlaceholder) {
      elEmptyPlaceholder.style.display = "none";
    }

    alertCount++;
    elAlertsCountBadge.textContent = `${alertCount} sự kiện ›`;

    const card = document.createElement("div");
    card.className = "alert-item";

    const eventId = alert.event_id || `evt_${Date.now()}_trk${alert.person_id}`;
    card.dataset.eventId = eventId;

    const timestamp = alert.timestamp ? new Date(alert.timestamp).toLocaleTimeString() : new Date().toLocaleTimeString();
    const action = alert.evidence?.primary_action || "Khẩn cấp";
    const score = typeof alert.risk_score === "number" ? alert.risk_score.toFixed(2) : "0.95";
    const lyingLabel = LYING_CONTEXT_LABELS[alert.evidence?.lying_context] || "";
    const warningSigns = [...new Set([...(alert.evidence?.warning_signs || []), ...(alert.evidence?.stroke_signs || [])])]
      .map((w) => LYING_CONTEXT_LABELS[w] || w).join(", ");
    const streamTitle = { fall: "TÉ NGÃ", stroke_warning: "DẤU HIỆU ĐỘT QUỴ / BỆNH CẤP" }[alert.event_type] || "";
    const onsetLabel = ONSET_REASON_LABELS[alert.evidence?.lying_onset_reason] || "";

    card.innerHTML = `
      <div class="alert-item-header">
        <div class="alert-item-title">
          <span>🚨</span>
          <span>${streamTitle ? `[${streamTitle}] ` : ""}${alert.camera_name ? `${escapeHtml(alert.camera_name)} · ` : ""}Đối tượng #${alert.person_id} — ${escapeHtml(action.toUpperCase())}</span>
        </div>
        <div class="alert-item-header-actions">
          <span class="alert-item-time">${timestamp}</span>
          <div class="alert-menu-wrapper">
            <button type="button" class="btn-alert-menu" title="Tùy chọn cảnh báo" aria-label="Tùy chọn cảnh báo">⋮</button>
            <div class="alert-menu-dropdown">
              <button type="button" class="dropdown-item btn-delete-single-perm" title="Xóa vĩnh viễn cảnh báo này">
                <span>🗑️</span> Xóa vĩnh viễn
              </button>
            </div>
          </div>
        </div>
      </div>
      <div class="alert-item-msg">
        ${alert.message || "Possible medical emergency / abnormal behavior detected. Please check the person."}
      </div>
      <div class="alert-item-meta">
        <span class="meta-field">Rủi ro: <strong>${score}</strong></span>
        <span class="meta-field">Vận tốc Vy: <strong>${alert.evidence?.vertical_velocity ?? "N/A"}</strong></span>
        <span class="meta-field">Góc thân: <strong>${alert.evidence?.torso_angle_deg ? alert.evidence.torso_angle_deg + "°" : "N/A"}</strong></span>
        <span class="meta-field">Bất động: <strong>${alert.evidence?.immobility_index ?? "N/A"}</strong></span>
        ${lyingLabel ? `<span class="meta-field">Ngữ cảnh: <strong>${escapeHtml(lyingLabel)}</strong></span>` : ""}
        ${warningSigns ? `<span class="meta-field">Dấu hiệu cảnh báo: <strong>${escapeHtml(warningSigns)}</strong></span>` : ""}
        ${onsetLabel ? `<span class="meta-field">Cách nằm xuống: <strong>${escapeHtml(onsetLabel)}</strong></span>` : ""}
      </div>
      ${alert.clip_path ? `
      <div class="alert-item-clip-row">
        <button type="button" class="btn-alert-clip">▶ Xem đoạn video té ngã</button>
      </div>` : ""}
    `;

    const btnClip = card.querySelector(".btn-alert-clip");
    if (btnClip) {
      btnClip.addEventListener("click", () => toggleAlertClip(card, eventId, btnClip));
    }

    // 3-Dots dropdown and individual permanent delete listener
    const btnMenu = card.querySelector(".btn-alert-menu");
    const dropdownMenu = card.querySelector(".alert-menu-dropdown");
    const btnDelete = card.querySelector(".btn-delete-single-perm");

    if (btnMenu && dropdownMenu) {
      btnMenu.addEventListener("click", (evt) => {
        evt.stopPropagation();
        // Close other open dropdowns
        document.querySelectorAll(".alert-menu-dropdown.show").forEach((el) => {
          if (el !== dropdownMenu) el.classList.remove("show");
        });
        document.querySelectorAll(".btn-alert-menu.active").forEach((el) => {
          if (el !== btnMenu) el.classList.remove("active");
        });

        dropdownMenu.classList.toggle("show");
        btnMenu.classList.toggle("active");
      });
    }

    if (btnDelete) {
      btnDelete.addEventListener("click", async (evt) => {
        evt.stopPropagation();
        dropdownMenu.classList.remove("show");
        btnMenu.classList.remove("active");

        const ok = window.confirm(`Bán có chắc muốn xóa vĩnh viễn cảnh báo này khỏi hệ thống không?`);
        if (!ok) return;

        await deleteSingleAlertPermanently(eventId, card);
      });
    }

    if (isNew) {
      elAlertsFeed.insertBefore(card, elAlertsFeed.firstChild);
    } else {
      elAlertsFeed.appendChild(card);
    }
  }

  // 5. Polling REST APIs for Telemetry & Health
  let lastConnState = null;
  async function pollStatus() {
    try {
      const res = await apiFetch(`${API_BASE}/status`);
      if (!res.ok) throw new Error("HTTP " + res.status);
      const data = await res.json();

      elBackendDot.className = "badge-dot dot-online";
      elBackendText.textContent = "API: Sẵn sàng";

      // Camera State
      isCameraRunning = Boolean(data.camera_active);
      elKpiFps.textContent = data.current_fps ? data.current_fps.toFixed(1) : "0.0";
      elKpiTracks.textContent = data.tracked_persons_count ?? 0;

      // Network / remote camera health (auto-reconnect with backoff runs on the server)
      const conn = data.camera_connection;
      if (conn && conn.state === "reconnecting") {
        const tries = conn.reconnect_attempts ? ` (lần thử ${conn.reconnect_attempts})` : "";
        showCamFeedback(`Mất kết nối camera ở xa, đang tự động kết nối lại${tries}...`, "error", 2500);
        lastConnState = "reconnecting";
      } else if (conn && conn.state === "connected" && lastConnState === "reconnecting") {
        showCamFeedback("Đã kết nối lại với camera.", "success", 3000);
        lastConnState = "connected";
      }

      // Update button labels
      if (isCameraRunning) {
        if (elBtnCamToggleTxt) elBtnCamToggleTxt.textContent = "DỪNG STREAM";
        if (elBtnCamToggleIcon) elBtnCamToggleIcon.textContent = "⏹️";
        if (elBtnCameraToggle) {
          elBtnCameraToggle.style.background = "rgba(239, 68, 68, 0.2)";
          elBtnCameraToggle.style.borderColor = "rgba(239, 68, 68, 0.4)";
          elBtnCameraToggle.style.color = "#f87171";
        }
        if (elOfflineOverlay) elOfflineOverlay.style.display = "none";
        
        // Camera connected: show its picture automatically
        if (!elVideoStream.getAttribute("src") || elVideoStream.src.indexOf("/stream/mjpeg") === -1) {
          elVideoStream.src = buildMjpegUrl();
        }
      } else {
        if (elBtnCamToggleTxt) elBtnCamToggleTxt.textContent = "BẬT STREAM";
        if (elBtnCamToggleIcon) elBtnCamToggleIcon.textContent = "▶️";
        if (elBtnCameraToggle) {
          elBtnCameraToggle.style.background = "";
          elBtnCameraToggle.style.borderColor = "";
          elBtnCameraToggle.style.color = "";
        }
        if (elOfflineOverlay) elOfflineOverlay.style.display = "flex";
      }

      // Update simulated or active persons in table
      updatePersonTable(data.tracked_persons_count);
    } catch (_) {
      elBackendDot.className = "badge-dot dot-offline";
      elBackendText.textContent = "API: Mất kết nối";
      elOfflineOverlay.style.display = "flex";
    }
  }

  async function pollHealth() {
    try {
      const res = await apiFetch(`${API_BASE}/health`);
      if (!res.ok) return;
      const data = await res.json();

      if (data.telemetry) {
        if (elTelemCpu) elTelemCpu.textContent = `${data.telemetry.cpu_percent ?? 0}%`;
        if (elTelemRam) elTelemRam.textContent = `${data.telemetry.ram_used_gb ?? 0} GB`;
        if (elTelemGpu) elTelemGpu.textContent = data.telemetry.gpu_available ? "CUDA (RTX 3050)" : "CPU Runtime";
      }
    } catch (_) {}
  }

  // 6. Update Risk Meter KPI
  function updateRiskKpi(score, stateText) {
    elKpiRisk.textContent = score.toFixed(2);
    elKpiRiskState.textContent = stateText || (score > 0.7 ? "NGUY CƠ CAO" : score > 0.4 ? "NGHI VẤN" : "BÌNH THƯỜNG");

    const pct = Math.min(100, Math.max(5, score * 100));
    elKpiRiskMeter.style.width = `${pct}%`;

    if (score > 0.7) {
      elKpiRisk.className = "kpi-value risk-red";
      elKpiRiskMeter.className = "kpi-meter-fill meter-red";
    } else if (score > 0.4) {
      elKpiRisk.className = "kpi-value risk-yellow";
      elKpiRiskMeter.className = "kpi-meter-fill meter-yellow";
    } else {
      elKpiRisk.className = "kpi-value risk-green";
      elKpiRiskMeter.className = "kpi-meter-fill meter-green";
    }
  }

  // Lying context from the risk engine (decision support only, never a diagnosis)
  const LYING_CONTEXT_LABELS = {
    lying: "Đang nằm",
    resting: "Nằm nghỉ / ngủ",
    using_phone: "Nằm dùng điện thoại",
    repositioning: "Trở mình",
    collapsed: "Đổ gục rồi nằm im",
    struggling: "Cố ngồi dậy nhiều lần không được",
    asymmetric_movement: "Chỉ cử động một bên người",
    still_on_floor: "Nằm im lâu trên sàn",
    staggering: "Đi loạng choạng, mất thăng bằng",
    abnormal_gesture: "Ôm đầu / ôm ngực / nôn",
    unsteady_before_fall: "Loạng choạng ngay trước khi ngã",
    collapse_from_sitting: "Đổ gục từ tư thế ngồi",
    convulsion: "Co giật (giật nhịp nhàng nhiều phần cơ thể)",
    after_convulsion: "Nằm sau cơn co giật",
  };

  // Lý do hệ thống kết luận cách người đó nằm xuống
  const ONSET_REASON_LABELS = {
    sat_down_first: "Ngồi xuống trước rồi nằm (chủ động)",
    paused_then_lowered: "Dừng lại rồi từ từ hạ người (chủ động)",
    slow_transition: "Hạ người chậm (chủ động)",
    awake_after_drop: "Nằm phịch xuống nhưng vẫn hoạt động bình thường",
    convulsion: "Co giật rồi nằm ra (không phải chủ động nằm)",
    unsteady_before: "Loạng choạng rồi ngã",
    collapse_from_sitting: "Đổ gục từ tư thế ngồi",
    fast_drop: "Rơi nhanh xuống",
    upright_to_floor: "Từ tư thế đứng xuống sàn trong vài giây",
    inherited: "Tiếp nối từ lần theo dõi trước",
  };

  // 7. Update Person Kinematics Table
  function updatePersonTable(trackCount) {
    if (!trackCount || trackCount === 0) {
      updateRiskKpi(0.05, "BÌNH THƯỜNG");
    }
    // The person table was removed from index.html; without this guard the TypeError
    // lands in pollStatus' catch and shows "API: Mất kết nối" over a running camera.
    if (!elPersonTableBody) return;

    if (!trackCount || trackCount === 0) {
      elPersonTableBody.innerHTML = `
        <tr>
          <td colspan="7" class="text-center text-muted">Không phát hiện người trong phòng (Phòng trống)</td>
        </tr>
      `;
      return;
    }

    let rowsHtml = "";
    for (let i = 1; i <= trackCount; i++) {
      rowsHtml += `
        <tr>
          <td><strong>#${i}</strong></td>
          <td><span class="action-badge badge-walking">walking</span></td>
          <td>12.4°</td>
          <td>0.48</td>
          <td>0.02 m/s</td>
          <td><span style="color: var(--status-normal);">0.08</span></td>
          <td><span style="color: var(--status-normal);">NORMAL</span></td>
        </tr>
      `;
    }
    elPersonTableBody.innerHTML = rowsHtml;
  }

  // 8. Fetch Historical Events on Load
  async function fetchHistoricalEvents() {
    try {
      const res = await apiFetch(`${API_BASE}/events?limit=20`);
      if (!res.ok) return;
      const events = await res.json();
      if (Array.isArray(events) && events.length > 0) {
        if (elEmptyPlaceholder) elEmptyPlaceholder.style.display = "none";
        events.forEach(evt => renderAlertCard(evt, false));
      }
    } catch (_) {}
  }

  // 9. Interactive UI Events
  // Stream error handler
  // The live <img> keeps retrying while the camera runs, so the picture comes back on its own
  // after a network drop or when the page loaded before the camera was ready.
  let streamRetryTimer = null;
  if (elVideoStream) {
    elVideoStream.addEventListener("error", () => {
      if (!isCameraRunning || !elVideoStream.getAttribute("src")) {
        if (elOfflineOverlay) elOfflineOverlay.style.display = "flex";
        return;
      }
      if (streamRetryTimer) return;
      streamRetryTimer = setTimeout(() => {
        streamRetryTimer = null;
        if (isCameraRunning) elVideoStream.src = buildMjpegUrl();
      }, 2000);
    });
    elVideoStream.addEventListener("load", () => {
      if (elOfflineOverlay) elOfflineOverlay.style.display = "none";
    });
  }

  // Toast feedback message helper
  let camToastTimer = null;
  function showCamFeedback(msg, type = "info", duration = 4500) {
    if (!elCamToastFeedback) return;
    if (camToastTimer) clearTimeout(camToastTimer);
    elCamToastFeedback.className = `cam-toast-feedback toast-${type}`;
    elCamToastFeedback.innerHTML = `<span>${type === "success" ? "✅" : (type === "error" ? "❌" : "ℹ️")}</span> <span>${msg}</span>`;
    elCamToastFeedback.style.display = "flex";
    if (duration > 0) {
      camToastTimer = setTimeout(() => {
        elCamToastFeedback.style.display = "none";
      }, duration);
    }
  }

  // Preset RTSP templates
  const RTSP_TEMPLATES = {
    custom: "",
    hikvision: "rtsp://admin:password123@192.168.1.100:554/Streaming/Channels/101",
    dahua: "rtsp://admin:password123@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0",
    yoosee: "rtsp://admin:123456@192.168.1.100:554/onvif1",
    tapo: "rtsp://admin:password123@192.168.1.100:554/stream1",
    ezviz: "rtsp://admin:VERIFICATION_CODE@192.168.1.100:554/h264/ch1/main/av_stream",
    ddns: "rtsp://tenmien-cua-ban.ddns.net:554/stream1",
    vpn: "rtsp://100.64.0.10:554/stream1",
    http_mjpeg: "http://192.168.1.50:8080/video",
  };

  function readRtspCredentials() {
    return {
      rtsp_username: elRtspUsernameInput && elRtspUsernameInput.value.trim() ? elRtspUsernameInput.value.trim() : undefined,
      rtsp_password: elRtspPasswordInput && elRtspPasswordInput.value ? elRtspPasswordInput.value : undefined,
    };
  }

  if (elBtnTestRtsp) {
    elBtnTestRtsp.addEventListener("click", async () => {
      const url = elRtspUrlInput ? elRtspUrlInput.value.trim() : "";
      if (!url) {
        showCamFeedback("Vui lòng nhập đường dẫn camera trước khi kiểm tra.", "error", 3500);
        return;
      }
      elBtnTestRtsp.disabled = true;
      elBtnTestRtsp.textContent = "⏳ Đang thử...";
      showCamFeedback("Đang thử kết nối tới camera (tối đa ~10 giây với camera ở xa)...", "info", 0);
      try {
        const res = await apiFetch(`${API_BASE}/camera/test`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rtsp_url: url, ...readRtspCredentials() }),
        });
        const data = await res.json();
        if (res.ok && data.ok) {
          showCamFeedback(`${data.message} (kết nối mất ${data.connect_ms} ms)`, "success", 6000);
        } else {
          // The backend explains the failure (own IP, RTSP off, wrong verification code, wrong path...)
          const msg = escapeHtml(data.message || data.detail || "Không kết nối được camera.");
          showCamFeedback(data.hint ? `${msg}<br>👉 ${escapeHtml(data.hint)}` : msg, "error", 15000);
        }
      } catch (e) {
        showCamFeedback("Lỗi kết nối tới Backend API: " + e.message, "error", 5000);
      } finally {
        elBtnTestRtsp.disabled = false;
        elBtnTestRtsp.textContent = "Kiểm tra";
      }
    });
  }

  // Put a discovered camera's IP (and RTSP port) into the URL, keeping the user's path / credentials
  function applyDiscoveredCamera(dev) {
    if (!elRtspUrlInput) return;
    if (elRtspPresetSelect && dev.preset && dev.preset !== "custom" && RTSP_TEMPLATES[dev.preset] !== undefined) {
      elRtspPresetSelect.value = dev.preset;
    }
    let url;
    if (!dev.rtsp_open && dev.http_port && dev.preset === "http_mjpeg") {
      url = `http://${dev.ip}:${dev.http_port}/video`;
    } else {
      const base = elRtspUrlInput.value.trim() || RTSP_TEMPLATES[dev.preset] || `rtsp://${dev.ip}:${dev.rtsp_port}/`;
      url = base.replace(/^(\w+:\/\/(?:[^@\/]*@)?)[^\/:?#]+(?::\d+)?/, `$1${dev.ip}:${dev.rtsp_port}`);
    }
    elRtspUrlInput.value = url;
    elRtspUrlInput.focus();
    if (dev.rtsp_open) {
      showCamFeedback(`Đã điền IP ${escapeHtml(dev.ip)}. Kiểm tra tài khoản / mật khẩu (EZVIZ: mã xác minh) rồi bấm "Kiểm tra".`, "success", 8000);
    } else {
      showCamFeedback(`Đã điền IP ${escapeHtml(dev.ip)}. ${escapeHtml(dev.status)}`, "info", 12000);
    }
  }

  async function discoverNetworkCameras() {
    if (!elBtnDiscoverCameras || !elCamDiscoverResults) return;
    elBtnDiscoverCameras.disabled = true;
    elBtnDiscoverCameras.textContent = "⏳ Đang dò mạng (~5 giây)...";
    elCamDiscoverResults.style.display = "none";
    try {
      const res = await apiFetch(`${API_BASE}/camera/discover`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || res.status);
      elCamDiscoverResults.innerHTML = "";
      const devices = Array.isArray(data.devices) ? data.devices : [];
      if (devices.length === 0) {
        const empty = document.createElement("div");
        empty.className = "cam-discover-empty";
        const nets = (data.networks || []).join(", ") || "không có mạng LAN";
        empty.textContent = `Không tìm thấy camera trong mạng ${nets}. Kiểm tra camera đã bật và cùng Wi-Fi với máy tính này, rồi dò lại.`;
        elCamDiscoverResults.appendChild(empty);
      }
      devices.forEach((dev) => {
        const item = document.createElement("button");
        item.type = "button";
        item.className = "cam-discover-item" + (dev.rtsp_open ? "" : " rtsp-off");
        item.innerHTML = `<span class="cam-discover-head">${escapeHtml(dev.ip)} · ${escapeHtml(dev.brand)}</span>`
          + `<span class="cam-discover-status">${dev.rtsp_open ? "✅" : "⚠️"} ${escapeHtml(dev.status)}</span>`;
        item.addEventListener("click", () => applyDiscoveredCamera(dev));
        elCamDiscoverResults.appendChild(item);
      });
      elCamDiscoverResults.style.display = "flex";
      if (devices.length > 0) {
        showCamFeedback(`Tìm thấy ${devices.length} thiết bị. Bấm vào một thiết bị để điền IP.`, "info", 5000);
      }
    } catch (e) {
      showCamFeedback("Không dò được mạng: " + escapeHtml(String(e.message || e)), "error", 6000);
    } finally {
      elBtnDiscoverCameras.disabled = false;
      elBtnDiscoverCameras.textContent = "📡 Dò camera trong mạng";
    }
  }

  if (elBtnDiscoverCameras) {
    elBtnDiscoverCameras.addEventListener("click", discoverNetworkCameras);
  }

  function webcamOptionLabel(dev) {
    let label = `${dev.device_index}: ${dev.name}`;
    if (dev.is_virtual) label += " — điện thoại / camera ảo";
    if (dev.is_active) label += " (đang dùng)";
    return label;
  }

  // List webcams by name (the server does not open any of them)
  async function fetchCameraDevices() {
    if (!elWebcamDeviceSelect) return;
    try {
      if (elBtnScanWebcams) {
        elBtnScanWebcams.disabled = true;
        elBtnScanWebcams.textContent = "⏳ Đang quét...";
      }
      const res = await apiFetch(`${API_BASE}/camera/devices`);
      if (!res.ok) return;
      const devices = await res.json();
      if (Array.isArray(devices) && devices.length > 0) {
        // Primary camera: every webcam except those already used by an extra camera
        elWebcamDeviceSelect.innerHTML = "";
        devices.forEach((dev) => {
          if (dev.in_use_by_extra && !dev.is_active) return;
          const opt = document.createElement("option");
          opt.value = dev.device_index;
          opt.textContent = webcamOptionLabel(dev);
          if (dev.is_active) opt.selected = true;
          elWebcamDeviceSelect.appendChild(opt);
        });
        // Extra camera: only webcams not used by the primary camera or another extra camera
        if (elExtraCamDevice && elExtraCamDevice.tagName === "SELECT") {
          elExtraCamDevice.innerHTML = "";
          devices.filter((dev) => !dev.is_active && !dev.in_use_by_extra).forEach((dev) => {
            const opt = document.createElement("option");
            opt.value = dev.device_index;
            opt.textContent = webcamOptionLabel(dev);
            elExtraCamDevice.appendChild(opt);
          });
          if (!elExtraCamDevice.options.length) {
            const opt = document.createElement("option");
            opt.value = "";
            opt.textContent = "Không còn webcam trống";
            elExtraCamDevice.appendChild(opt);
          }
        }
      }
    } catch (err) {
      console.warn("[Dashboard] Lỗi khi quét danh sách camera:", err);
    } finally {
      if (elBtnScanWebcams) {
        elBtnScanWebcams.disabled = false;
        elBtnScanWebcams.textContent = "🔍 Quét";
      }
    }
  }

  // Fetch Active Camera Status & Information
  async function fetchCameraInfo() {
    try {
      const res = await apiFetch(`${API_BASE}/camera/info`);
      if (!res.ok) return;
      const info = await res.json();
      if (info) {
        const srcType = info.source_type || "webcam";
        if (elCameraTypeSelect) {
          elCameraTypeSelect.value = srcType === "synthetic" ? "video" : srcType;
        }
        handleCameraTypeChange();

        if (info.device_index !== undefined && elWebcamDeviceSelect) {
          elWebcamDeviceSelect.value = String(info.device_index);
        }
        if (info.rtsp_url && elRtspUrlInput) {
          elRtspUrlInput.value = info.rtsp_url;
        }

        const camName = info.name || (srcType === "webcam" ? `Webcam ${info.device_index ?? 0}` : (srcType === "rtsp" ? "RTSP Camera" : "Video File"));
        if (elCamActiveNameTxt) elCamActiveNameTxt.textContent = camName;
        if (elHudCamName) elHudCamName.textContent = `CAM: ${camName.toUpperCase()}`;
      }
    } catch (_) {}
  }

  // Handle Camera Type Switch (Webcam / RTSP / Video)
  function handleCameraTypeChange() {
    const selectedType = elCameraTypeSelect ? elCameraTypeSelect.value : "webcam";
    if (elGroupWebcamDevice) elGroupWebcamDevice.style.display = selectedType === "webcam" ? "flex" : "none";
    if (elGroupRtspPreset) elGroupRtspPreset.style.display = selectedType === "rtsp" ? "flex" : "none";
    if (elGroupRtspUrlRow) elGroupRtspUrlRow.style.display = selectedType === "rtsp" ? "flex" : "none";
    if (elGroupVideoPreset) elGroupVideoPreset.style.display = selectedType === "video" ? "flex" : "none";
    if (elGroupVideoCustomRow) {
      elGroupVideoCustomRow.style.display = (selectedType === "video" && elVideoPresetSelect && elVideoPresetSelect.value === "custom") ? "flex" : "none";
    }
  }

  if (elCameraTypeSelect) {
    elCameraTypeSelect.addEventListener("change", handleCameraTypeChange);
  }

  if (elRtspPresetSelect) {
    elRtspPresetSelect.addEventListener("change", (e) => {
      const presetKey = e.target.value;
      if (RTSP_TEMPLATES[presetKey] && elRtspUrlInput) {
        elRtspUrlInput.value = RTSP_TEMPLATES[presetKey];
        elRtspUrlInput.focus();
      }
    });
  }

  if (elVideoPresetSelect) {
    elVideoPresetSelect.addEventListener("change", (e) => {
      if (elGroupVideoCustomRow) {
        elGroupVideoCustomRow.style.display = e.target.value === "custom" ? "flex" : "none";
      }
    });
  }

  if (elBtnScanWebcams) {
    elBtnScanWebcams.addEventListener("click", () => {
      fetchCameraDevices();
      showCamFeedback("Đang quét các cổng webcam trên máy tính...", "info", 2000);
    });
  }

  // Apply & Switch Camera
  async function applyCameraSwitch(saveDefault = false) {
    const srcType = elCameraTypeSelect ? elCameraTypeSelect.value : "webcam";
    let devIndex = 0;
    let rtspUrl = "";
    let videoPath = "";

    if (srcType === "webcam") {
      devIndex = parseInt(elWebcamDeviceSelect ? elWebcamDeviceSelect.value : "0", 10) || 0;
    } else if (srcType === "rtsp") {
      rtspUrl = elRtspUrlInput ? elRtspUrlInput.value.trim() : "";
      if (!rtspUrl) {
        showCamFeedback("Vui lòng nhập đường dẫn camera (ví dụ: rtsp://192.168.1.100:554/stream1 hoặc rtsp://tenmien.ddns.net:554/stream1)", "error", 4000);
        if (elRtspUrlInput) elRtspUrlInput.focus();
        return;
      }
    } else if (srcType === "video") {
      const preset = elVideoPresetSelect ? elVideoPresetSelect.value : "data/videos/sample_adl_fall.mp4";
      if (preset === "custom") {
        videoPath = elCustomVideoPathInput ? elCustomVideoPathInput.value.trim() : "";
        if (!videoPath) {
          showCamFeedback("Vui lòng nhập đường dẫn tệp video MP4 cục bộ", "error", 3000);
          return;
        }
      } else {
        videoPath = preset;
      }
    }

    // Set UI loading state
    if (elBtnApplyCamera) elBtnApplyCamera.disabled = true;
    if (elCamSpinner) elCamSpinner.style.display = "inline-block";
    if (elBtnApplyTxt) elBtnApplyTxt.textContent = "ĐANG KẾT NỐI CAMERA...";
    showCamFeedback("Đang giải phóng camera cũ và kết nối nguồn mới...", "info", 0);

    try {
      const res = await apiFetch(`${API_BASE}/camera/switch`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          source_type: srcType,
          device_index: devIndex,
          rtsp_url: rtspUrl || undefined,
          ...(srcType === "rtsp" ? readRtspCredentials() : {}),
          video_path: videoPath || undefined,
          save_as_default: saveDefault,
        }),
      });

      const data = await res.json();

      if (res.ok && data.status === "success") {
        isCameraRunning = true;
        if (elOfflineOverlay) elOfflineOverlay.style.display = "none";
        showCamFeedback(data.message || "Đã chuyển đổi camera thành công!", "success", 4000);
        fetchCameraDevices();

        // Update active badge & HUD
        const newName = data.camera_info?.name || "Camera";
        if (elCamActiveNameTxt) elCamActiveNameTxt.textContent = newName;
        if (elHudCamName) elHudCamName.textContent = `CAM: ${newName.toUpperCase()}`;

        // Reload MJPEG stream with cachebuster
        setTimeout(() => {
          if (elVideoStream) {
            elVideoStream.src = buildMjpegUrl();
          }
          pollStatus();
        }, 400);

      } else {
        const msg = data.detail || data.message || "Không thể kết nối đến camera được chọn.";
        showCamFeedback(escapeHtml(msg), "error", 12000);
        pollStatus();
      }
    } catch (e) {
      showCamFeedback("Lỗi kết nối tới Backend API: " + e.message, "error", 5000);
      pollStatus();
    } finally {
      if (elBtnApplyCamera) elBtnApplyCamera.disabled = false;
      if (elCamSpinner) elCamSpinner.style.display = "none";
      if (elBtnApplyTxt) elBtnApplyTxt.textContent = "🔄 ÁP DỤNG & CHUYỂN CAMERA";
    }
  }

  if (elBtnApplyCamera) {
    elBtnApplyCamera.addEventListener("click", () => applyCameraSwitch(false));
  }

  if (elBtnSaveCameraDefault) {
    elBtnSaveCameraDefault.addEventListener("click", () => applyCameraSwitch(true));
  }

  // Toggle Camera Start / Stop
  async function toggleCamera() {
    if (isCameraRunning) {
      try {
        await apiFetch(`${API_BASE}/camera/stop`, { method: "POST" });
        isCameraRunning = false;
        if (elVideoStream) elVideoStream.removeAttribute("src");
        if (elOfflineOverlay) elOfflineOverlay.style.display = "flex";
        showCamFeedback("Đã tạm dừng luồng camera giám sát.", "info", 2500);
        pollStatus();
      } catch (e) {
        showCamFeedback("Lỗi khi dừng camera: " + e.message, "error", 3000);
      }
    } else {
      await applyCameraSwitch(false);
    }
  }

  if (elBtnCameraToggle) elBtnCameraToggle.addEventListener("click", toggleCamera);

  // Refresh stream
  if (elBtnRefreshStream) {
    elBtnRefreshStream.addEventListener("click", () => {
      if (elVideoStream) {
        elVideoStream.src = buildMjpegUrl();
        showCamFeedback("Đã tải lại luồng hiển thị video.", "info", 2000);
      }
    });
  }

  // Dismiss Emergency Banner & Silence Alarm
  elBtnDismissAlert.addEventListener("click", () => {
    stopContinuousAlarm();
    clearEmergencyVisuals();
    elEmergencyBar.style.display = "none";
    updateRiskKpi(0.05, "BÌNH THƯỜNG");
  });

  // Xóa đơn lẻ vĩnh viễn 1 cảnh báo
  async function deleteSingleAlertPermanently(eventId, cardElement) {
    if (eventId) {
      try {
        await apiFetch(`${API_BASE}/events/${encodeURIComponent(eventId)}`, { method: "DELETE" });
      } catch (err) {
        console.warn("[Dashboard] Lỗi kết nối khi xóa sự kiện trên server:", err);
      }
    }

    if (cardElement) {
      cardElement.style.transition = "all 0.25s ease-out";
      cardElement.style.opacity = "0";
      cardElement.style.transform = "translateX(25px)";
      setTimeout(() => {
        if (cardElement.parentNode) {
          cardElement.parentNode.removeChild(cardElement);
        }
      }, 250);
    }

    alertCount = Math.max(0, alertCount - 1);
    elAlertsCountBadge.textContent = alertCount > 0 ? `${alertCount} sự kiện ›` : "0 sự kiện ›";

    if (alertCount === 0) {
      if (elEmptyPlaceholder) {
        elEmptyPlaceholder.style.display = "flex";
      }
      stopContinuousAlarm();
      clearEmergencyVisuals();
      if (elEmergencyBar) elEmergencyBar.style.display = "none";
      updateRiskKpi(0.05, "BÌNH THƯỜNG");
    }
  }

  // Chức năng 1: Xóa tạm thời (Ẩn khỏi màn hình hiện tại & tắt chuông, không xóa dữ liệu máy chủ)
  function clearTempAlerts() {
    elAlertsFeed.innerHTML = "";
    elAlertsFeed.appendChild(elEmptyPlaceholder);
    elEmptyPlaceholder.style.display = "flex";
    alertCount = 0;
    elAlertsCountBadge.textContent = "0 sự kiện ›";

    stopContinuousAlarm();
    clearEmergencyVisuals();
    if (elEmergencyBar) elEmergencyBar.style.display = "none";
    updateRiskKpi(0.05, "BÌNH THƯỜNG");

    console.log("[Dashboard] Danh sách cảnh báo đã được xóa tạm thời khỏi màn hình.");
  }

  // Chức năng 2: Xóa vĩnh viễn (Xóa toàn bộ trên giao diện và xóa vĩnh viễn dữ liệu trên Backend API)
  async function clearPermanentAlerts() {
    const confirmClear = window.confirm(
      "CẢNH BÁO XÓA VĨNH VIỄN:\nToàn bộ dữ liệu lịch sử cảnh báo sẽ bị xóa sạch khỏi hệ thống máy chủ và không thể khôi phục.\nBạn có chắc chắn muốn xóa không?"
    );
    if (!confirmClear) return;

    elAlertsFeed.innerHTML = "";
    elAlertsFeed.appendChild(elEmptyPlaceholder);
    elEmptyPlaceholder.style.display = "flex";
    alertCount = 0;
    elAlertsCountBadge.textContent = "0 sự kiện ›";

    stopContinuousAlarm();
    clearEmergencyVisuals();
    if (elEmergencyBar) elEmergencyBar.style.display = "none";
    updateRiskKpi(0.05, "BÌNH THƯỜNG");

    try {
      await apiFetch(`${API_BASE}/events/clear`, { method: "POST" });
      console.log("[Dashboard] Toàn bộ dữ liệu cảnh báo đã được xóa vĩnh viễn trên Server.");
    } catch (err) {
      console.warn("[Dashboard] Lỗi khi gửi yêu cầu xóa vĩnh viễn lên server:", err);
    }
  }

  if (elBtnClearTemp) {
    elBtnClearTemp.addEventListener("click", () => {
      clearTempAlerts();
      if (elAlertsMenuDropdown) elAlertsMenuDropdown.classList.remove("show");
      if (elBtnAlertsMenuToggle) elBtnAlertsMenuToggle.classList.remove("active");
    });
  }
  if (elBtnClearPermanent) {
    elBtnClearPermanent.addEventListener("click", async () => {
      await clearPermanentAlerts();
      if (elAlertsMenuDropdown) elAlertsMenuDropdown.classList.remove("show");
      if (elBtnAlertsMenuToggle) elBtnAlertsMenuToggle.classList.remove("active");
    });
  }

  // Header 3-Dots Dropdown Menu for Alerts Card
  const elBtnAlertsMenuToggle = document.getElementById("btn-alerts-menu-toggle");
  const elAlertsMenuDropdown = document.getElementById("alerts-menu-dropdown");

  if (elBtnAlertsMenuToggle && elAlertsMenuDropdown) {
    elBtnAlertsMenuToggle.addEventListener("click", (evt) => {
      evt.stopPropagation();
      // Close other item dropdowns
      document.querySelectorAll(".alert-menu-dropdown.show").forEach((el) => el.classList.remove("show"));
      document.querySelectorAll(".btn-alert-menu.active").forEach((el) => el.classList.remove("active"));
      document.querySelectorAll(".bubble-popover-menu.show").forEach((el) => el.classList.remove("show"));
      document.querySelectorAll(".btn-bubble-dots.active").forEach((el) => el.classList.remove("active"));

      elAlertsMenuDropdown.classList.toggle("show");
      elBtnAlertsMenuToggle.classList.toggle("active");
    });
  }

  // Camera 3-Dots Bubble Popover Menu (Image 2 style)
  const elBtnCamMenuToggle = document.getElementById("btn-cam-menu-toggle");
  const elCamPopoverMenu = document.getElementById("cam-popover-menu");

  if (elBtnCamMenuToggle && elCamPopoverMenu) {
    elBtnCamMenuToggle.addEventListener("click", (evt) => {
      evt.stopPropagation();
      // Close alert dropdown
      if (elAlertsMenuDropdown) elAlertsMenuDropdown.classList.remove("show");
      if (elBtnAlertsMenuToggle) elBtnAlertsMenuToggle.classList.remove("active");

      const isOpen = elCamPopoverMenu.classList.toggle("show");
      elBtnCamMenuToggle.classList.toggle("active", isOpen);
    });

    // Prevent accidental closing when interacting with controls inside popover
    elCamPopoverMenu.addEventListener("click", (evt) => {
      evt.stopPropagation();
    });
  }

  // Snapshot Capture Button (Inside 3-Dots Popover)
  const elBtnSnapshot = document.getElementById("btn-snapshot");
  if (elBtnSnapshot) {
    elBtnSnapshot.addEventListener("click", () => {
      const videoImg = document.getElementById("live-video-stream");
      if (!videoImg || !videoImg.naturalWidth) {
        showCamFeedback("Chưa có luồng video hoạt động để chụp ảnh.", "error", 2500);
        return;
      }
      try {
        const canvas = document.createElement("canvas");
        canvas.width = videoImg.naturalWidth;
        canvas.height = videoImg.naturalHeight;
        const ctx = canvas.getContext("2d");
        ctx.drawImage(videoImg, 0, 0);
        const link = document.createElement("a");
        const timestamp = new Date().toISOString().replace(/[:.]/g, "-");
        link.download = `camera_snapshot_${timestamp}.png`;
        link.href = canvas.toDataURL("image/png");
        link.click();
        showCamFeedback("📸 Đã chụp và tải ảnh snapshot về máy!", "success", 2500);
      } catch (err) {
        console.warn("[Dashboard] Snapshot error:", err);
        showCamFeedback("Đã lưu khung hình snapshot!", "info", 2500);
      }
    });
  }

  // Đóng toàn bộ dropdown / popover 3 chấm khi click ra ngoài
  document.addEventListener("click", () => {
    document.querySelectorAll(".alert-menu-dropdown.show").forEach((el) => el.classList.remove("show"));
    document.querySelectorAll(".chrome-menu-dropdown.show").forEach((el) => el.classList.remove("show"));
    document.querySelectorAll(".bubble-popover-menu.show").forEach((el) => el.classList.remove("show"));
    document.querySelectorAll(".btn-alert-menu.active").forEach((el) => el.classList.remove("active"));
    document.querySelectorAll(".btn-chrome-more.active").forEach((el) => el.classList.remove("active"));
    document.querySelectorAll(".btn-bubble-dots.active").forEach((el) => el.classList.remove("active"));
  });

  const elSoundIconIndicator = document.getElementById("sound-icon-indicator");
  const elSoundBadgeStatus = document.getElementById("sound-badge-status");

  function updateSoundUi() {
    if (elSoundIconIndicator) {
      elSoundIconIndicator.textContent = soundEnabled ? "🔊" : "🔇";
    }
    if (elSoundBadgeStatus) {
      elSoundBadgeStatus.textContent = soundEnabled ? "BẬT" : "TẮT";
      elSoundBadgeStatus.className = soundEnabled ? "item-badge-status status-on" : "item-badge-status status-off";
    }
  }

  // Nút Test Chuông Cảnh Báo
  if (elBtnTestChime) {
    elBtnTestChime.addEventListener("click", () => {
      initAudioContext();
      if (!soundEnabled) {
        soundEnabled = true;
        updateSoundUi();
      }
      console.log("[Dashboard] Testing emergency alarm chime over speaker...");
      triggerAlarmChime("fall");
    });
  }

  // Sound Toggle
  if (elBtnSoundToggle) {
    elBtnSoundToggle.addEventListener("click", () => {
      soundEnabled = !soundEnabled;
      if (!soundEnabled) {
        stopContinuousAlarm();
      } else {
        initAudioContext();
        playBellChimeTone(880, 0.35, 0.4); // Tiếng chuông nhẹ xác nhận đã bật loa
      }
      updateSoundUi();
    });
  }

  // Fullscreen toggle
  elBtnFullscreen.addEventListener("click", () => {
    const container = document.getElementById("video-container");
    if (!document.fullscreenElement) {
      container.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen().catch(() => {});
    }
  });

  // ==============================================================================
  // 9. SETTINGS MODAL CONTROLLER (BẢNG CÀI ĐẶT HỆ THỐNG GỌN GÀNG 1-CLICK)
  // ==============================================================================
  const elBtnOpenSettings = document.getElementById("btn-open-settings");
  const elBtnCloseSettings = document.getElementById("btn-close-settings");
  const elSettingsOverlay = document.getElementById("settings-modal-overlay");
  const elToggleAutostart = document.getElementById("toggle-autostart");
  const elAutostartPill = document.getElementById("autostart-status-pill");
  const elAutostartFeedback = document.getElementById("autostart-feedback");
  const elBtnCheckCamera = document.getElementById("btn-check-camera");
  const elBtnOpenWinCamera = document.getElementById("btn-open-win-camera-settings");
  const elCameraCheckFeedback = document.getElementById("camera-check-feedback");
  const elToggleFaceBlur = document.getElementById("toggle-face-blur");
  const elBtnModalShutdown = document.getElementById("btn-modal-shutdown");

  function openSettingsModal() {
    if (elSettingsOverlay) elSettingsOverlay.style.display = "flex";
    fetchAutostartStatus();
    fetchTelegramSettings();
    fetchUpdateStatus();
  }

  // ---- Cập nhật phiên bản (GitHub Releases) ----
  const elUpdatePill = document.getElementById("update-status-pill");
  const elUpdateCheck = document.getElementById("btn-update-check");
  const elUpdateDownload = document.getElementById("btn-update-download");
  const elUpdateInstall = document.getElementById("btn-update-install");
  const elUpdateProgress = document.getElementById("update-progress");
  const elUpdateProgressBar = document.getElementById("update-progress-bar");
  const elUpdateNotes = document.getElementById("update-notes");
  const elUpdateFeedback = document.getElementById("update-feedback");
  const elAppVersionText = document.getElementById("app-version-text");
  let updatePollTimer = null;
  let installArmedUntil = 0;

  function showUpdateFeedback(text, kind) {
    if (!elUpdateFeedback) return;
    elUpdateFeedback.style.display = text ? "block" : "none";
    elUpdateFeedback.className = `settings-feedback ${kind === "error" ? "feedback-error" : "feedback-success"}`;
    elUpdateFeedback.textContent = text || "";
  }

  function formatMb(bytes) {
    return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  }

  async function readApiError(res) {
    try {
      const data = await res.json();
      return data.detail || `HTTP ${res.status}`;
    } catch (_) {
      return `HTTP ${res.status}`;
    }
  }

  function renderUpdateStatus(st) {
    if (!st) return;
    const latest = st.latest;
    if (elAppVersionText) elAppVersionText.textContent = `v${st.current_version}`;
    const elSidebarVersion = document.getElementById("sidebar-version-tag");
    if (elSidebarVersion) elSidebarVersion.textContent = `Phiên bản ${st.current_version}`;
    if (elUpdatePill) {
      elUpdatePill.textContent = st.update_available && latest ? `Có bản mới v${latest.version}` : `v${st.current_version}`;
      elUpdatePill.className = `badge-status-pill ${st.update_available ? "pill-on" : "pill-off"}`;
    }

    const downloading = st.state === "downloading";
    const ready = st.state === "ready";
    if (elUpdateDownload) elUpdateDownload.style.display = st.update_available && !downloading && !ready && st.state !== "installing" ? "inline-flex" : "none";
    if (elUpdateInstall) elUpdateInstall.style.display = ready && st.can_install ? "inline-flex" : "none";
    if (elUpdateProgress) elUpdateProgress.style.display = downloading || ready ? "block" : "none";
    if (elUpdateProgressBar) {
      const pct = st.total_bytes > 0 ? Math.min(100, (st.downloaded_bytes / st.total_bytes) * 100) : (ready ? 100 : 0);
      elUpdateProgressBar.style.width = `${pct.toFixed(0)}%`;
    }
    if (elUpdateNotes) {
      const show = st.update_available && latest;
      elUpdateNotes.style.display = show ? "block" : "none";
      if (show) {
        const date = latest.published_at ? new Date(latest.published_at).toLocaleDateString() : "";
        elUpdateNotes.textContent = `Phiên bản ${latest.version}${date ? ` (phát hành ${date})` : ""}\n\n${latest.notes || "Không có ghi chú phát hành."}`;
      }
    }

    if (st.state === "error" && st.error) {
      showUpdateFeedback(`⚠ ${st.error}`, "error");
    } else if (downloading) {
      const total = st.total_bytes ? ` / ${formatMb(st.total_bytes)}` : "";
      showUpdateFeedback(`Đang tải bộ cài: ${formatMb(st.downloaded_bytes)}${total}...`, "success");
    } else if (ready) {
      showUpdateFeedback(st.can_install
        ? "✓ Đã tải xong và kiểm tra bộ cài. Bấm \"Cài đặt & khởi động lại\" khi sẵn sàng."
        : "✓ Đã tải xong. Đang chạy từ mã nguồn nên không tự cài được: hãy cập nhật bằng git pull, hoặc cài bằng bộ cài trên GitHub.", "success");
    } else if (st.state === "installing") {
      showUpdateFeedback("Đang cài bản mới... Hệ thống sẽ tự mở lại sau khoảng 1 phút, sau đó tải lại trang này.", "success");
    } else if (st.check_error) {
      showUpdateFeedback(`⚠ ${st.check_error}`, "error");
    } else if (st.checked_at && !st.update_available) {
      showUpdateFeedback(`✓ Bạn đang dùng phiên bản mới nhất (v${st.current_version}).`, "success");
    } else {
      showUpdateFeedback("", "success");
    }

    if (downloading && !updatePollTimer) {
      updatePollTimer = setInterval(fetchUpdateStatus, 1000);
    } else if (!downloading && updatePollTimer) {
      clearInterval(updatePollTimer);
      updatePollTimer = null;
    }
  }

  async function fetchUpdateStatus() {
    try {
      const res = await apiFetch(`${API_BASE}/api/update/status`);
      if (res.ok) renderUpdateStatus(await res.json());
    } catch (_) {}
  }

  async function postUpdate(action, busyText) {
    showUpdateFeedback(busyText, "success");
    try {
      const res = await apiFetch(`${API_BASE}/api/update/${action}`, { method: "POST" });
      if (!res.ok) {
        showUpdateFeedback(`⚠ ${await readApiError(res)}`, "error");
        return null;
      }
      const st = await res.json();
      renderUpdateStatus(st);
      return st;
    } catch (e) {
      showUpdateFeedback(`⚠ Không kết nối được máy chủ: ${e.message}`, "error");
      return null;
    }
  }

  if (elUpdateCheck) elUpdateCheck.addEventListener("click", () => postUpdate("check", "Đang kiểm tra bản mới trên GitHub..."));
  if (elUpdateDownload) elUpdateDownload.addEventListener("click", () => postUpdate("download", "Bắt đầu tải bộ cài..."));
  if (elUpdateInstall) {
    elUpdateInstall.addEventListener("click", async () => {
      // Xác nhận ngay trên trang (bấm lần 2 trong 5 giây), không dùng hộp thoại của trình duyệt
      if (Date.now() > installArmedUntil) {
        installArmedUntil = Date.now() + 5000;
        showUpdateFeedback("Hệ thống sẽ TẠM DỪNG GIÁM SÁT khoảng 1 phút để cài đặt. Bấm \"Cài đặt & khởi động lại\" thêm lần nữa để xác nhận.", "error");
        return;
      }
      installArmedUntil = 0;
      const st = await postUpdate("install", "Đang khởi chạy bộ cài...");
      if (st) setTimeout(() => location.reload(), 90000);
    });
  }

  fetchUpdateStatus();

  function closeSettingsModal() {
    if (elSettingsOverlay) elSettingsOverlay.style.display = "none";
  }

  if (elBtnOpenSettings) elBtnOpenSettings.addEventListener("click", openSettingsModal);
  if (elBtnCloseSettings) elBtnCloseSettings.addEventListener("click", closeSettingsModal);
  if (elSettingsOverlay) {
    elSettingsOverlay.addEventListener("click", (e) => {
      if (e.target === elSettingsOverlay) closeSettingsModal();
    });
  }

  // 1. Khởi động cùng Windows
  async function fetchAutostartStatus() {
    try {
      const res = await apiFetch(`${API_BASE}/api/settings/autostart`);
      if (res.ok) {
        const data = await res.json();
        if (elToggleAutostart) elToggleAutostart.checked = !!data.enabled;
        updateAutostartPill(!!data.enabled);
      }
    } catch (_) {}
  }

  function updateAutostartPill(enabled) {
    if (!elAutostartPill) return;
    if (enabled) {
      elAutostartPill.textContent = "Đang BẬT";
      elAutostartPill.className = "badge-status-pill pill-on";
    } else {
      elAutostartPill.textContent = "Đang TẮT";
      elAutostartPill.className = "badge-status-pill pill-off";
    }
  }

  if (elToggleAutostart) {
    elToggleAutostart.addEventListener("change", async () => {
      const enabled = elToggleAutostart.checked;
      try {
        const res = await apiFetch(`${API_BASE}/api/settings/autostart?enable=${enabled}`, { method: "POST" });
        const data = await res.json();
        updateAutostartPill(data.enabled);
        if (elAutostartFeedback) {
          elAutostartFeedback.style.display = "block";
          elAutostartFeedback.className = "settings-feedback feedback-success";
          elAutostartFeedback.textContent = data.message || (enabled ? "Đã bật tự động chạy ngầm cùng Windows!" : "Đã tắt khởi động cùng Windows!");
          setTimeout(() => { if (elAutostartFeedback) elAutostartFeedback.style.display = "none"; }, 4000);
        }
      } catch (err) {
        if (elAutostartFeedback) {
          elAutostartFeedback.style.display = "block";
          elAutostartFeedback.className = "settings-feedback feedback-error";
          elAutostartFeedback.textContent = "Lỗi khi cập nhật cài đặt khởi động.";
        }
      }
    });
  }

  // 2. Quyền & Kiểm tra Camera
  if (elBtnCheckCamera) {
    elBtnCheckCamera.addEventListener("click", async () => {
      elBtnCheckCamera.disabled = true;
      elBtnCheckCamera.innerHTML = "<span>⏳</span> Đang kiểm tra...";
      try {
        const res = await apiFetch(`${API_BASE}/api/settings/check-camera`, { method: "POST" });
        const data = await res.json();
        if (elCameraCheckFeedback) {
          elCameraCheckFeedback.style.display = "block";
          if (data.status) {
            elCameraCheckFeedback.className = "settings-feedback feedback-success";
            elCameraCheckFeedback.textContent = `✓ ${data.message} (Cổng tìm thấy: ${data.working_devices?.join(", ") || "0"})`;
          } else {
            elCameraCheckFeedback.className = "settings-feedback feedback-error";
            elCameraCheckFeedback.textContent = `⚠ ${data.message}`;
          }
        }
      } catch (err) {
        if (elCameraCheckFeedback) {
          elCameraCheckFeedback.style.display = "block";
          elCameraCheckFeedback.className = "settings-feedback feedback-error";
          elCameraCheckFeedback.textContent = "Không thể kết nối API để kiểm tra camera.";
        }
      } finally {
        elBtnCheckCamera.disabled = false;
        elBtnCheckCamera.innerHTML = "<span>🔍</span> Kiểm tra Camera";
      }
    });
  }

  if (elBtnOpenWinCamera) {
    elBtnOpenWinCamera.addEventListener("click", async () => {
      try {
        await apiFetch(`${API_BASE}/api/settings/open-camera-settings`, { method: "POST" });
      } catch (_) {}
    });
  }

  // 3. Làm mờ khuôn mặt (Face Blur Toggle)
  if (elToggleFaceBlur) {
    elToggleFaceBlur.addEventListener("change", async () => {
      const blur = elToggleFaceBlur.checked;
      try {
        await apiFetch(`${API_BASE}/api/camera/blur?enabled=${blur}`, { method: "POST" });
        const elBlurStatus = document.getElementById("hud-blur-status");
        if (elBlurStatus) {
          elBlurStatus.textContent = blur ? "MẶT: ĐÃ LÀM MỜ" : "MẶT: RÕ NÉT (KHÔNG LÀM MỜ)";
        }
      } catch (_) {}
    });
  }

  // 4. Thông báo Telegram
  const elToggleTelegram = document.getElementById("toggle-telegram");
  const elTelegramPill = document.getElementById("telegram-status-pill");
  const elTelegramToken = document.getElementById("telegram-token-input");
  const elTelegramChat = document.getElementById("telegram-chat-input");
  const elTelegramChatSelect = document.getElementById("telegram-chat-select");
  const elBtnTelegramDetect = document.getElementById("btn-telegram-detect");
  const elBtnTelegramSave = document.getElementById("btn-telegram-save");
  const elBtnTelegramTest = document.getElementById("btn-telegram-test");
  const elTelegramFeedback = document.getElementById("telegram-feedback");

  function showTelegramFeedback(ok, message) {
    if (!elTelegramFeedback) return;
    elTelegramFeedback.style.display = "block";
    elTelegramFeedback.className = `settings-feedback ${ok ? "feedback-success" : "feedback-error"}`;
    elTelegramFeedback.textContent = `${ok ? "✓" : "⚠"} ${message}`;
  }

  function updateTelegramPill(settings) {
    if (!elTelegramPill) return;
    if (settings.ready) {
      elTelegramPill.textContent = "Đang BẬT";
      elTelegramPill.className = "badge-status-pill pill-on";
    } else {
      elTelegramPill.textContent = settings.has_token ? "Đang TẮT" : "Chưa cài đặt";
      elTelegramPill.className = "badge-status-pill pill-off";
    }
  }

  function applyTelegramSettings(settings) {
    if (elToggleTelegram) elToggleTelegram.checked = !!settings.enabled;
    if (elTelegramToken) {
      elTelegramToken.value = "";
      elTelegramToken.placeholder = settings.has_token
        ? `Đã lưu (${settings.token_masked}) — bỏ trống để giữ nguyên`
        : "123456789:AAH...";
    }
    if (elTelegramChat) elTelegramChat.value = settings.chat_id || "";
    updateTelegramPill(settings);
  }

  async function fetchTelegramSettings() {
    try {
      const res = await apiFetch(`${API_BASE}/api/settings/telegram`);
      if (res.ok) applyTelegramSettings(await res.json());
    } catch (_) {}
  }

  async function postTelegram(path, body) {
    const res = await apiFetch(`${API_BASE}/api/settings/telegram${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(data.detail || `Lỗi máy chủ (HTTP ${res.status})`);
    return data;
  }

  async function withBusyButton(button, busyLabel, action) {
    if (!button) return;
    const original = button.innerHTML;
    button.disabled = true;
    button.innerHTML = `<span>⏳</span> ${busyLabel}`;
    try {
      await action();
    } catch (err) {
      showTelegramFeedback(false, err.message || "Không thể kết nối API.");
    } finally {
      button.disabled = false;
      button.innerHTML = original;
    }
  }

  function telegramFormValues() {
    return {
      bot_token: elTelegramToken ? elTelegramToken.value.trim() : "",
      chat_id: elTelegramChat ? elTelegramChat.value.trim() : "",
    };
  }

  if (elBtnTelegramDetect) {
    elBtnTelegramDetect.addEventListener("click", () => withBusyButton(elBtnTelegramDetect, "Đang tìm...", async () => {
      const data = await postTelegram("/detect-chat", { bot_token: telegramFormValues().bot_token });
      if (elTelegramChatSelect) {
        elTelegramChatSelect.innerHTML = "";
        elTelegramChatSelect.style.display = data.chats.length > 1 ? "block" : "none";
        data.chats.forEach((chat) => {
          const opt = document.createElement("option");
          opt.value = chat.chat_id;
          opt.textContent = `${chat.name} (${chat.chat_id})`;
          elTelegramChatSelect.appendChild(opt);
        });
      }
      if (data.chats.length && elTelegramChat) elTelegramChat.value = data.chats[0].chat_id;
      showTelegramFeedback(data.status === "success", data.message);
    }));
  }

  if (elTelegramChatSelect) {
    elTelegramChatSelect.addEventListener("change", () => {
      if (elTelegramChat) elTelegramChat.value = elTelegramChatSelect.value;
    });
  }

  if (elBtnTelegramSave) {
    elBtnTelegramSave.addEventListener("click", () => withBusyButton(elBtnTelegramSave, "Đang lưu...", async () => {
      const enabled = elToggleTelegram ? elToggleTelegram.checked : true;
      const data = await postTelegram("", { ...telegramFormValues(), enabled });
      applyTelegramSettings(data.settings);
      showTelegramFeedback(true, data.message);
    }));
  }

  if (elBtnTelegramTest) {
    elBtnTelegramTest.addEventListener("click", () => withBusyButton(elBtnTelegramTest, "Đang gửi...", async () => {
      const data = await postTelegram("/test", telegramFormValues());
      showTelegramFeedback(true, data.message);
    }));
  }

  // 5. Tắt hệ thống an toàn
  if (elBtnModalShutdown) {
    elBtnModalShutdown.addEventListener("click", async () => {
      if (!confirm("Bạn có chắc chắn muốn TẮT hệ thống AI giám sát và dừng máy chủ không?")) return;
      elBtnModalShutdown.disabled = true;
      elBtnModalShutdown.textContent = "Đang tắt hệ thống...";
      try {
        await apiFetch(`${API_BASE}/api/system/shutdown`, { method: "POST" });
        alert("Hệ thống đã dừng an toàn! Bạn có thể đóng cửa sổ trình duyệt này.");
        window.close();
      } catch (_) {
        alert("Đã gửi lệnh tắt hệ thống.");
      }
    });
  }

  // 6. Kết nối các tương tác trên giao diện Compact mới
  const btnQuickFull = document.getElementById("btn-quick-fullscreen");
  if (btnQuickFull) {
    btnQuickFull.addEventListener("click", () => {
      if (elBtnFullscreen) elBtnFullscreen.click();
    });
  }

  const sidebarSettings = document.getElementById("sidebar-btn-settings");
  if (sidebarSettings && elBtnOpenSettings) {
    sidebarSettings.addEventListener("click", (e) => {
      e.preventDefault();
      elBtnOpenSettings.click();
    });
  }

  const btnProcessInc = document.getElementById("btn-process-incident");
  if (btnProcessInc) {
    btnProcessInc.addEventListener("click", () => {
      const tag = document.getElementById("incident-status-tag");
      if (tag) {
        tag.textContent = "Đã xử lý";
        tag.style.background = "#ecfdf5";
        tag.style.borderColor = "#a7f3d0";
        tag.style.color = "#059669";
      }
      stopContinuousAlarm();
      clearEmergencyVisuals();
      if (elEmergencyBar) elEmergencyBar.style.display = "none";
      updateRiskKpi(0.05, "Bình thường");
    });
  }

  const btnViewInc = document.getElementById("btn-view-incident-detail");
  if (btnViewInc) {
    btnViewInc.addEventListener("click", () => {
      const historyCard = document.getElementById("alerts-history-card");
      if (historyCard) historyCard.scrollIntoView({ behavior: "smooth" });
    });
  }

  document.querySelectorAll(".sidebar-link").forEach(link => {
    link.addEventListener("click", (e) => {
      if (link.id === "sidebar-btn-settings") return;
      document.querySelectorAll(".sidebar-link").forEach(l => l.classList.remove("active"));
      link.classList.add("active");
    });
  });


  // ============================================================================
  // CAMERA KHÁC (ĐA CAMERA): mỗi camera thêm có AI, theo dõi và cảnh báo riêng
  // ============================================================================
  const elExtraCamGrid = document.getElementById("extra-cam-grid");
  const elCamList = document.getElementById("cam-list");
  const elExtraCamEmpty = document.getElementById("extra-cam-empty");
  const elExtraCamCount = document.getElementById("extra-cam-count");
  const elExtraCamForm = document.getElementById("extra-cam-form");
  const elBtnExtraCamToggleForm = document.getElementById("btn-extra-cam-toggle-form");
  const elBtnExtraCamCancel = document.getElementById("btn-extra-cam-cancel");
  const elBtnExtraCamAdd = document.getElementById("btn-extra-cam-add");
  const elBtnExtraCamTest = document.getElementById("btn-extra-cam-test");
  const elExtraCamName = document.getElementById("extra-cam-name");
  const elExtraCamType = document.getElementById("extra-cam-type");
  const elExtraCamUrl = document.getElementById("extra-cam-url");
  const elExtraCamUser = document.getElementById("extra-cam-user");
  const elExtraCamPass = document.getElementById("extra-cam-pass");
  const elExtraCamDevice = document.getElementById("extra-cam-device");
  const elExtraCamRtspRow = document.getElementById("extra-cam-rtsp-row");
  const elExtraCamWebcamRow = document.getElementById("extra-cam-webcam-row");
  const elExtraCamFeedback = document.getElementById("extra-cam-feedback");

  const extraCamTiles = new Map();         // camera id -> { tile, img, offline, meta }
  const extraCamAlertUntil = new Map();    // camera id -> time until which the tile flashes red
  let extraCamFeedbackTimer = null;

  function escapeHtml(value) {
    return String(value ?? "").replace(/[&<>"']/g, ch => (
      { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
    ));
  }

  // Same scheme as buildMjpegUrl (cache-buster only)
  function buildExtraCamStreamUrl(camId) {
    const token = getSavedToken();
    const tokenParam = token ? `&token=${encodeURIComponent(token)}` : "";
    return `${API_BASE}/api/cameras/${encodeURIComponent(camId)}/stream?t=${Date.now()}${tokenParam}`;
  }

  function showExtraCamFeedback(msg, type = "info", duration = 5000) {
    if (!elExtraCamFeedback) return;
    if (extraCamFeedbackTimer) clearTimeout(extraCamFeedbackTimer);
    elExtraCamFeedback.className = `cam-toast-feedback toast-${type}`;
    elExtraCamFeedback.innerHTML = `<span>${type === "success" ? "✅" : (type === "error" ? "❌" : "ℹ️")}</span> <span>${escapeHtml(msg)}</span>`;
    elExtraCamFeedback.style.display = "flex";
    if (duration > 0) {
      extraCamFeedbackTimer = setTimeout(() => { elExtraCamFeedback.style.display = "none"; }, duration);
    }
  }

  function createExtraCamTile(cam) {
    const tile = document.createElement("div");
    tile.className = "extra-cam-tile";
    tile.dataset.cameraId = cam.id;
    tile.innerHTML = `
      <div class="extra-cam-video">
        <img alt="Camera ${escapeHtml(cam.name)}">
        <div class="extra-cam-offline">Đang kết nối…</div>
      </div>
      <div class="extra-cam-footer">
        <span class="extra-cam-name" title="${escapeHtml(cam.name)}">${escapeHtml(cam.name)}</span>
        <span class="extra-cam-meta"></span>
        <button type="button" class="btn-extra-cam-remove" title="Xóa camera này">🗑️ Xóa</button>
      </div>
    `;
    const entry = {
      tile,
      img: tile.querySelector("img"),
      offline: tile.querySelector(".extra-cam-offline"),
      meta: tile.querySelector(".extra-cam-meta"),
    };
    // A dropped MJPEG connection is re-opened on the next poll
    entry.img.addEventListener("error", () => entry.img.removeAttribute("src"));
    tile.querySelector(".btn-extra-cam-remove").addEventListener("click", () => removeExtraCamera(cam.id, cam.name));
    return entry;
  }

  function updateExtraCamTile(entry, cam) {
    const flashing = cam.emergency || (extraCamAlertUntil.get(cam.id) || 0) > Date.now();
    entry.tile.classList.toggle("is-emergency", Boolean(flashing));

    if (cam.online) {
      entry.offline.style.display = "none";
    } else {
      const conn = cam.connection;
      let text = "Đang kết nối…";
      if (cam.enabled === false) text = "Camera đang tắt";
      else if (conn && conn.state === "reconnecting") {
        text = `Mất kết nối, đang tự kết nối lại${conn.reconnect_attempts ? ` (lần thử ${conn.reconnect_attempts})` : ""}…`;
      } else if (cam.last_frame_age_s != null) text = "Mất tín hiệu camera";
      entry.offline.textContent = text;
      entry.offline.style.display = "flex";
    }
    if (!entry.img.getAttribute("src") && cam.enabled !== false) {
      entry.img.src = buildExtraCamStreamUrl(cam.id);
    }

    const dotClass = flashing ? "alert" : (cam.online ? "on" : "");
    const label = cam.online ? `${cam.tracked_persons ?? 0} người` : "Ngoại tuyến";
    entry.meta.innerHTML = `<span class="dot ${dotClass}"></span>${escapeHtml(label)}`;
  }

  function describeCameraSource(cam) {
    if (cam.source_type === "webcam") return `Webcam ${cam.device_index ?? 0}`;
    if (cam.source_type === "video") return "Tệp video";
    if (cam.source_type === "synthetic") return "Giả lập";
    return cam.rtsp_url || "Camera IP";
  }

  function viewExtraCamera(camId) {
    const entry = extraCamTiles.get(camId);
    if (!entry) return;
    const box = entry.tile.querySelector(".extra-cam-video");
    entry.tile.scrollIntoView({ behavior: "smooth", block: "center" });
    if (box && box.requestFullscreen) box.requestFullscreen().catch(() => {});
  }

  // One row per camera: the main camera (changed in the ⋯ menu) and every added camera (view / delete)
  function renderCameraList(primary, cameras) {
    if (!elCamList) return;
    const rows = [];
    if (primary) {
      rows.push({ id: "main", name: primary.name || "Camera chính", source: describeCameraSource(primary),
                  online: primary.online, main: true });
    }
    cameras.forEach(cam => rows.push({ id: cam.id, name: cam.name, source: describeCameraSource(cam),
                                       online: cam.online, enabled: cam.enabled }));
    elCamList.innerHTML = "";
    rows.forEach(row => {
      const item = document.createElement("div");
      item.className = "cam-list-row";
      const status = row.online ? "Đang hoạt động" : (row.enabled === false ? "Đang tắt" : "Ngoại tuyến");
      item.innerHTML = `
        <span class="dot ${row.online ? "on" : ""}"></span>
        <div class="cam-list-info">
          <span class="cam-list-name">${escapeHtml(row.name)}${row.main ? ' <span class="cam-list-tag">Chính</span>' : ""}</span>
          <span class="cam-list-source" title="${escapeHtml(row.source)}">${escapeHtml(row.source)} · ${status}</span>
        </div>
        <div class="cam-list-actions"></div>`;
      const actions = item.querySelector(".cam-list-actions");
      const viewBtn = document.createElement("button");
      viewBtn.type = "button";
      viewBtn.className = "btn-cam-action";
      viewBtn.textContent = "Xem";
      viewBtn.addEventListener("click", () => {
        if (row.main) {
          document.getElementById("video-container")?.scrollIntoView({ behavior: "smooth", block: "center" });
        } else {
          viewExtraCamera(row.id);
        }
      });
      actions.appendChild(viewBtn);
      if (!row.main) {
        const delBtn = document.createElement("button");
        delBtn.type = "button";
        delBtn.className = "btn-extra-cam-remove";
        delBtn.textContent = "🗑️ Xóa";
        delBtn.addEventListener("click", () => removeExtraCamera(row.id, row.name));
        actions.appendChild(delBtn);
      }
      elCamList.appendChild(item);
    });
  }

  async function fetchExtraCameras() {
    if (!elExtraCamGrid) return;
    try {
      const res = await apiFetch(`${API_BASE}/api/cameras`);
      if (!res.ok) return;
      const data = await res.json();
      const cameras = data.cameras || [];
      const maxCams = data.max_cameras || 0;

      if (elExtraCamCount) elExtraCamCount.textContent = `${cameras.length}/${maxCams}`;
      if (elBtnExtraCamToggleForm) elBtnExtraCamToggleForm.disabled = maxCams > 0 && cameras.length >= maxCams;
      if (elExtraCamEmpty) elExtraCamEmpty.style.display = cameras.length ? "none" : "block";

      const elCamsOnline = document.getElementById("stat-cameras-online");
      if (elCamsOnline) {
        const online = (data.primary && data.primary.online ? 1 : 0) + cameras.filter(c => c.online).length;
        elCamsOnline.textContent = online;
      }

      renderCameraList(data.primary, cameras);

      const seen = new Set();
      cameras.forEach(cam => {
        seen.add(cam.id);
        let entry = extraCamTiles.get(cam.id);
        if (!entry) {
          entry = createExtraCamTile(cam);
          extraCamTiles.set(cam.id, entry);
          elExtraCamGrid.appendChild(entry.tile);
        }
        updateExtraCamTile(entry, cam);
      });
      for (const [id, entry] of extraCamTiles) {
        if (!seen.has(id)) {
          entry.img.removeAttribute("src");
          entry.tile.remove();
          extraCamTiles.delete(id);
        }
      }
    } catch (e) {
      // API offline: pollStatus already shows the backend state
    }
  }

  async function removeExtraCamera(camId, camName) {
    if (!window.confirm(`Xóa camera "${camName}" khỏi danh sách giám sát?`)) return;
    try {
      const res = await apiFetch(`${API_BASE}/api/cameras/${encodeURIComponent(camId)}`, { method: "DELETE" });
      if (!res.ok) {
        const data = await res.json().catch(() => ({}));
        alert(data.detail || "Không xóa được camera.");
      }
      fetchCameraDevices();
    } finally {
      fetchExtraCameras();
    }
  }

  function readExtraCamForm() {
    const sourceType = elExtraCamType ? elExtraCamType.value : "rtsp";
    const body = { name: elExtraCamName ? elExtraCamName.value.trim() : "", source_type: sourceType };
    if (sourceType === "webcam") {
      const idx = parseInt(elExtraCamDevice ? elExtraCamDevice.value : "", 10);
      if (Number.isInteger(idx)) body.device_index = idx;
    } else {
      body.rtsp_url = elExtraCamUrl ? elExtraCamUrl.value.trim() : "";
      if (elExtraCamUser && elExtraCamUser.value.trim()) body.rtsp_username = elExtraCamUser.value.trim();
      if (elExtraCamPass && elExtraCamPass.value) body.rtsp_password = elExtraCamPass.value;
    }
    return body;
  }

  function setExtraCamFormVisible(visible) {
    if (!elExtraCamForm) return;
    elExtraCamForm.style.display = visible ? "block" : "none";
    if (!visible) {
      elExtraCamForm.reset();
      handleExtraCamTypeChange();
      if (elExtraCamFeedback) elExtraCamFeedback.style.display = "none";
    } else {
      fetchCameraDevices();
      if (elExtraCamName) elExtraCamName.focus();
    }
  }

  function handleExtraCamTypeChange() {
    const isWebcam = elExtraCamType && elExtraCamType.value === "webcam";
    if (elExtraCamRtspRow) elExtraCamRtspRow.style.display = isWebcam ? "none" : "flex";
    if (elExtraCamWebcamRow) elExtraCamWebcamRow.style.display = isWebcam ? "flex" : "none";
  }

  if (elExtraCamType) elExtraCamType.addEventListener("change", handleExtraCamTypeChange);
  if (elBtnExtraCamToggleForm) {
    elBtnExtraCamToggleForm.addEventListener("click", () => {
      setExtraCamFormVisible(elExtraCamForm && elExtraCamForm.style.display === "none");
    });
  }
  if (elBtnExtraCamCancel) elBtnExtraCamCancel.addEventListener("click", () => setExtraCamFormVisible(false));

  if (elBtnExtraCamTest) {
    elBtnExtraCamTest.addEventListener("click", async () => {
      const body = readExtraCamForm();
      if (!body.rtsp_url) {
        showExtraCamFeedback("Vui lòng nhập đường dẫn camera trước khi kiểm tra.", "error");
        return;
      }
      elBtnExtraCamTest.disabled = true;
      elBtnExtraCamTest.textContent = "⏳ Đang thử...";
      showExtraCamFeedback("Đang thử kết nối tới camera (tối đa ~10 giây với camera ở xa)...", "info", 0);
      try {
        const res = await apiFetch(`${API_BASE}/camera/test`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rtsp_url: body.rtsp_url, rtsp_username: body.rtsp_username, rtsp_password: body.rtsp_password }),
        });
        const data = await res.json();
        if (res.ok && data.ok) showExtraCamFeedback(`${data.message} (kết nối mất ${data.connect_ms} ms)`, "success", 6000);
        else showExtraCamFeedback([data.message || data.detail || "Không kết nối được camera.", data.hint].filter(Boolean).join(" 👉 "), "error", 15000);
      } catch (e) {
        showExtraCamFeedback("Lỗi kết nối tới Backend API: " + e.message, "error");
      } finally {
        elBtnExtraCamTest.disabled = false;
        elBtnExtraCamTest.textContent = "Kiểm tra";
      }
    });
  }

  if (elExtraCamForm) {
    elExtraCamForm.addEventListener("submit", async (ev) => {
      ev.preventDefault();
      const body = readExtraCamForm();
      if (body.source_type === "rtsp" && !body.rtsp_url) {
        showExtraCamFeedback("Vui lòng nhập đường dẫn camera.", "error");
        return;
      }
      if (body.source_type === "webcam" && body.device_index === undefined) {
        showExtraCamFeedback("Không còn webcam trống để thêm.", "error");
        return;
      }
      elBtnExtraCamAdd.disabled = true;
      elBtnExtraCamAdd.textContent = "⏳ ĐANG THÊM...";
      try {
        const res = await apiFetch(`${API_BASE}/api/cameras`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await res.json().catch(() => ({}));
        if (res.ok) {
          setExtraCamFormVisible(false);
          fetchExtraCameras();
          fetchCameraDevices();
        } else {
          showExtraCamFeedback(data.detail || "Không thêm được camera.", "error", 8000);
        }
      } catch (e) {
        showExtraCamFeedback("Lỗi kết nối tới Backend API: " + e.message, "error");
      } finally {
        elBtnExtraCamAdd.disabled = false;
        elBtnExtraCamAdd.textContent = "THÊM CAMERA";
      }
    });
  }

  // Alert from an extra camera: flash that camera's tile for 10 s
  function flashExtraCameraTile(camId) {
    extraCamAlertUntil.set(camId, Date.now() + 10000);
    const entry = extraCamTiles.get(camId);
    if (entry) {
      entry.tile.classList.add("is-emergency");
      entry.tile.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
    setTimeout(fetchExtraCameras, 10050);
  }

  // ========================================================================
  // 10. Authentication Flow & Dashboard Initialization
  // ========================================================================
  function initDashboard() {
    if (isDashboardInitialized) return;
    isDashboardInitialized = true;

    // Attach stream feed with auth token
    if (elVideoStream && (!elVideoStream.src || elVideoStream.src === window.location.href)) {
      elVideoStream.src = buildMjpegUrl();
    }

    connectAlertsWebSocket();
    fetchHistoricalEvents();
    fetchCameraDevices();
    fetchCameraInfo();
    pollStatus();
    pollHealth();
    fetchAutostartStatus();

    // Polling intervals
    setInterval(pollStatus, 1500);
    setInterval(pollHealth, 4000);
    fetchExtraCameras();
    setInterval(fetchExtraCameras, 3000);

    console.log("[Dashboard] Initialized successfully. Connected to API:", API_BASE);
  }

  // status: /api/auth/status payload. The first admin can only be created on the machine
  // running the system (or with `python -m src.auth.cli create-admin` on a server).
  function showAuthModal(status) {
    if (!elAuthModalBackdrop) return;
    const setupPendingElsewhere = Boolean(status && status.setup_required && !status.setup_allowed_here);
    isSetupMode = Boolean(status && status.setup_allowed_here);

    if (isSetupMode) {
      if (elAuthModalTitle) elAuthModalTitle.textContent = "Thiết Lập Bảo Vệ (Lần Đầu)";
      if (elAuthModalSubtitle) {
        elAuthModalSubtitle.textContent =
          "Tạo tài khoản quản trị một lần duy nhất (mật khẩu tối thiểu 8 ký tự) để các thiết bị khác trong mạng có thể đăng nhập xem camera.";
      }
      if (elAuthSubmitTxt) elAuthSubmitTxt.textContent = "Tạo Tài Khoản & Ghi Nhớ Thiết Bị";
    } else {
      if (elAuthModalTitle) elAuthModalTitle.textContent = "Xác Thực Bảo Mật Mạng Nội Bộ";
      if (elAuthModalSubtitle) {
        elAuthModalSubtitle.textContent =
          "Thiết bị này đang kết nối từ mạng nội bộ (Wi-Fi/LAN). Vui lòng đăng nhập để xem camera và nhận cảnh báo an toàn. Tài khoản sẽ được ghi nhớ trực tiếp trên máy này.";
      }
      if (elAuthSubmitTxt) elAuthSubmitTxt.textContent = "Đăng Nhập & Ghi Nhớ Thiết Bị";
    }

    if (elAuthError) {
      elAuthError.style.display = "none";
      elAuthError.textContent = "";
      if (setupPendingElsewhere) {
        elAuthError.textContent = "Chưa có tài khoản quản trị. Hãy tạo tài khoản trên chính máy tính chạy hệ thống (mở http://localhost:8000 trên máy đó), rồi đăng nhập tại đây.";
        elAuthError.style.display = "block";
      }
    }
    elAuthModalBackdrop.style.display = "flex";
    if (elAuthUsername) elAuthUsername.focus();
  }

  function handleAuthRequired() {
    fetch(`${API_BASE}/api/auth/status`)
      .then((r) => r.json())
      .then((data) => {
        if (!data.can_access_without_login) {
          showAuthModal(data);
        }
      })
      .catch(() => {});
  }

  if (elAuthForm) {
    elAuthForm.addEventListener("submit", async (e) => {
      e.preventDefault();
      const username = elAuthUsername ? elAuthUsername.value.trim() : "";
      const password = elAuthPassword ? elAuthPassword.value.trim() : "";

      if (!username || !password) {
        if (elAuthError) {
          elAuthError.textContent = "Vui lòng nhập đầy đủ tên đăng nhập và mật khẩu.";
          elAuthError.style.display = "block";
        }
        return;
      }

      if (elBtnAuthSubmit) elBtnAuthSubmit.disabled = true;
      if (elAuthSubmitTxt) elAuthSubmitTxt.textContent = "Đang xử lý...";
      if (elAuthError) elAuthError.style.display = "none";

      const endpoint = isSetupMode ? "/api/auth/setup" : "/api/auth/login";
      try {
        const res = await fetch(`${API_BASE}${endpoint}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ username, password }),
        });
        const json = await res.json();
        if (res.ok && json.access_token) {
          saveAuthToken(json.access_token);
          saveRefreshToken(json.refresh_token);
          if (elAuthModalBackdrop) elAuthModalBackdrop.style.display = "none";
          // Reload camera stream and initialize dashboard
          if (elVideoStream) elVideoStream.src = buildMjpegUrl();
          initDashboard();
        } else {
          if (elAuthError) {
            elAuthError.textContent = json.detail || "Đăng nhập không thành công. Vui lòng kiểm tra lại.";
            elAuthError.style.display = "block";
          }
        }
      } catch (err) {
        if (elAuthError) {
          elAuthError.textContent = "Lỗi kết nối đến máy chủ: " + err.message;
          elAuthError.style.display = "block";
        }
      } finally {
        if (elBtnAuthSubmit) elBtnAuthSubmit.disabled = false;
        if (elAuthSubmitTxt) {
          elAuthSubmitTxt.textContent = isSetupMode
            ? "Tạo Tài Khoản & Ghi Nhớ Thiết Bị"
            : "Đăng Nhập & Ghi Nhớ Thiết Bị";
        }
      }
    });
  }

  // Start sequence with auth check:
  // - Localhost (127.0.0.1): zero friction, opens and runs immediately!
  // - LAN/Wi-Fi: checks the saved token (renewed with the refresh token); if not present, shows modal.
  async function checkAuthAndInit() {
    try {
      const res = await fetch(`${API_BASE}/api/auth/status`);
      const data = await res.json();

      if (data.can_access_without_login) {
        if (elAuthModalBackdrop) elAuthModalBackdrop.style.display = "none";
        initDashboard();
        return;
      }

      const token = getSavedToken();
      if (token || getRefreshToken()) {
        let verifyRes = await fetch(`${API_BASE}/api/auth/me`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        if (verifyRes.status === 401 && await refreshAccessToken()) {
          verifyRes = await fetch(`${API_BASE}/api/auth/me`, {
            headers: { Authorization: `Bearer ${getSavedToken()}` },
          });
        }
        if (verifyRes.ok) {
          if (elAuthModalBackdrop) elAuthModalBackdrop.style.display = "none";
          initDashboard();
          return;
        } else {
          clearAuthToken();
        }
      }

      showAuthModal(data);
    } catch (e) {
      console.warn("[Dashboard] Auth check error, initializing anyway:", e);
      initDashboard();
    }
  }

  checkAuthAndInit();
})();
