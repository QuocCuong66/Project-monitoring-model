/**
 * ELDERLY AI MONITOR — FRONTEND DASHBOARD CONTROLLER
 * Real-time WebSocket connection, REST telemetry polling, and MJPEG video streaming.
 */

(() => {
  "use strict";

  // 1. API Configuration & Auto-Discovery
  const isFileOrigin = window.location.protocol === "file:";
  const API_BASE = isFileOrigin ? "http://127.0.0.1:8000" : window.location.origin;
  const WS_URL = API_BASE.replace(/^http/, "ws") + "/ws/alerts";

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
  const elBtnStartOverlay = document.getElementById("btn-start-camera-overlay");

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
  function updateClock() {
    const now = new Date();
    elClock.textContent = now.toTimeString().split(" ")[0];
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
      wsConnection = new WebSocket(WS_URL);

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

      wsConnection.onclose = () => {
        elWsDot.className = "badge-dot dot-offline";
        elWsText.textContent = "WebSocket: Mất kết nối";
        scheduleWsReconnect();
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

    // Phân loại: Nguy cơ té ngã vs Hành vi bất thường
    const isFallRisk = (
      primaryAction === "falling" ||
      primaryAction === "stumbling" ||
      primaryAction === "lying" ||
      primaryAction === "immobile" ||
      riskScore >= 0.70 ||
      (alert.evidence?.drop_severity_score && alert.evidence.drop_severity_score > 0.35)
    );

    // 1. Show flash emergency banner
    elEmergencyBar.style.display = "block";
    const alertTypeLabel = isFallRisk ? "NGUY CƠ TÉ NGÃ KHẨN CẤP" : "HÀNH VI BẤT THƯỜNG";
    elAlertBarDesc.textContent = `[${alertTypeLabel}] Đối tượng ID: #${personId} — ${primaryAction.toUpperCase()} (Điểm rủi ro: ${riskScore.toFixed(2)}). Possible medical emergency / abnormal behavior detected. Please check the person.`;

    // 2. Hiện khung viền màu đỏ chớp nháy cảnh báo & Floating badge trên video
    if (elVideoContainer) elVideoContainer.classList.add("fall-emergency-border");
    if (elVideoCard) elVideoCard.classList.add("fall-emergency-border");
    if (elVideoFallBadge) {
      elVideoFallBadge.style.display = "flex";
      if (elVideoFallText) {
        elVideoFallText.textContent = isFallRisk
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
    const elFallDemoBox = document.getElementById("video-fall-box-demo");

    if (elIncTs) elIncTs.textContent = `${new Date().toLocaleTimeString()} - ${new Date().toLocaleDateString('vi-VN')}`;
    if (elIncConf) elIncConf.textContent = `${(riskScore * 100).toFixed(0)}%`;
    if (elIncStatus) {
      elIncStatus.textContent = "Đang xử lý";
      elIncStatus.style.background = "#fef2f2";
      elIncStatus.style.borderColor = "#fecaca";
      elIncStatus.style.color = "#dc2626";
    }
    if (elFallDemoBox) {
      elFallDemoBox.style.display = isFallRisk ? "block" : "none";
    }

    // 5. Thêm dòng vào Bảng Lịch Sử Cảnh Báo
    const elAlertsTbody = document.getElementById("alerts-table-body");
    if (elAlertsTbody) {
      const tr = document.createElement("tr");
      tr.className = "alert-row-highlight";
      tr.innerHTML = `
        <td><span class="id-tag">#${String(alertCount + 1).padStart(3, '0')}</span></td>
        <td>${new Date().toLocaleTimeString()}</td>
        <td><span class="badge-type-danger">${isFallRisk ? "Té ngã" : "Bất thường"}</span></td>
        <td>Camera 01</td>
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
    const url = `${API_BASE}/events/${encodeURIComponent(eventId)}/clip`;
    btn.disabled = true;
    try {
      const res = await fetch(`${API_BASE}/events/${encodeURIComponent(eventId)}`);
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

    card.innerHTML = `
      <div class="alert-item-header">
        <div class="alert-item-title">
          <span>🚨</span>
          <span>Đối tượng #${alert.person_id} — ${action.toUpperCase()}</span>
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

        const ok = window.confirm(`Bạn có chắc muốn xóa vĩnh viễn cảnh báo này khỏi hệ thống không?`);
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
      const res = await fetch(`${API_BASE}/status`);
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
        
        // Ensure stream URL is attached
        const streamUrl = `${API_BASE}/stream/mjpeg`;
        if (!elVideoStream.src || elVideoStream.src.indexOf("/stream/mjpeg") === -1) {
          elVideoStream.src = streamUrl;
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

    } catch (err) {
      elBackendDot.className = "badge-dot dot-offline";
      elBackendText.textContent = "API: Mất kết nối";
      elOfflineOverlay.style.display = "flex";
    }
  }

  async function pollHealth() {
    try {
      const res = await fetch(`${API_BASE}/health`);
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

  // 7. Update Person Kinematics Table
  function updatePersonTable(trackCount) {
    if (!trackCount || trackCount === 0) {
      elPersonTableBody.innerHTML = `
        <tr>
          <td colspan="7" class="text-center text-muted">Không phát hiện người trong phòng (Phòng trống)</td>
        </tr>
      `;
      updateRiskKpi(0.05, "BÌNH THƯỜNG");
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
      const res = await fetch(`${API_BASE}/events?limit=20`);
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
  window.handleStreamError = () => {
    if (isCameraRunning) {
      // Auto-reattach with cache buster after brief network pause
      setTimeout(() => {
        elVideoStream.src = `${API_BASE}/stream/mjpeg?t=${Date.now()}`;
      }, 800);
    } else {
      elOfflineOverlay.style.display = "flex";
    }
  };

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
        const res = await fetch(`${API_BASE}/camera/test`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ rtsp_url: url, ...readRtspCredentials() }),
        });
        const data = await res.json();
        if (res.ok && data.ok) {
          showCamFeedback(`${data.message} (kết nối mất ${data.connect_ms} ms)`, "success", 6000);
        } else {
          showCamFeedback(data.message || data.detail || "Không kết nối được camera.", "error", 7000);
        }
      } catch (e) {
        showCamFeedback("Lỗi kết nối tới Backend API: " + e.message, "error", 5000);
      } finally {
        elBtnTestRtsp.disabled = false;
        elBtnTestRtsp.textContent = "Kiểm tra";
      }
    });
  }

  // Fetch Available Webcam Devices on Machine
  async function fetchCameraDevices() {
    if (!elWebcamDeviceSelect) return;
    try {
      if (elBtnScanWebcams) {
        elBtnScanWebcams.disabled = true;
        elBtnScanWebcams.textContent = "⏳ Đang quét...";
      }
      const res = await fetch(`${API_BASE}/camera/devices`);
      if (!res.ok) return;
      const devices = await res.json();
      if (Array.isArray(devices) && devices.length > 0) {
        elWebcamDeviceSelect.innerHTML = "";
        devices.forEach((dev) => {
          const opt = document.createElement("option");
          opt.value = dev.device_index;
          opt.textContent = `${dev.name} (${dev.resolution || "720p"})`;
          if (dev.is_active) opt.selected = true;
          elWebcamDeviceSelect.appendChild(opt);
        });
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
      const res = await fetch(`${API_BASE}/camera/info`);
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
      const res = await fetch(`${API_BASE}/camera/switch`, {
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

        // Update active badge & HUD
        const newName = data.camera_info?.name || "Camera";
        if (elCamActiveNameTxt) elCamActiveNameTxt.textContent = newName;
        if (elHudCamName) elHudCamName.textContent = `CAM: ${newName.toUpperCase()}`;

        // Reload MJPEG stream with cachebuster
        setTimeout(() => {
          if (elVideoStream) {
            elVideoStream.src = `${API_BASE}/stream/mjpeg?t=${Date.now()}`;
          }
          pollStatus();
        }, 400);

      } else {
        const msg = data.detail || data.message || "Không thể kết nối đến camera được chọn.";
        showCamFeedback(msg, "error", 6000);
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
        await fetch(`${API_BASE}/camera/stop`, { method: "POST" });
        isCameraRunning = false;
        if (elVideoStream) elVideoStream.src = "";
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
  if (elBtnStartOverlay) elBtnStartOverlay.addEventListener("click", () => applyCameraSwitch(false));

  // Refresh stream
  if (elBtnRefreshStream) {
    elBtnRefreshStream.addEventListener("click", () => {
      if (elVideoStream) {
        elVideoStream.src = `${API_BASE}/stream/mjpeg?t=${Date.now()}`;
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
        await fetch(`${API_BASE}/events/${encodeURIComponent(eventId)}`, { method: "DELETE" });
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
      await fetch(`${API_BASE}/events/clear`, { method: "POST" });
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
  }

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
      const res = await fetch(`${API_BASE}/api/settings/autostart`);
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
        const res = await fetch(`${API_BASE}/api/settings/autostart?enable=${enabled}`, { method: "POST" });
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
        const res = await fetch(`${API_BASE}/api/settings/check-camera`, { method: "POST" });
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
        await fetch(`${API_BASE}/api/settings/open-camera-settings`, { method: "POST" });
      } catch (_) {}
    });
  }

  // 3. Làm mờ khuôn mặt (Face Blur Toggle)
  if (elToggleFaceBlur) {
    elToggleFaceBlur.addEventListener("change", async () => {
      const blur = elToggleFaceBlur.checked;
      try {
        await fetch(`${API_BASE}/api/camera/blur?enabled=${blur}`, { method: "POST" });
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
      const res = await fetch(`${API_BASE}/api/settings/telegram`);
      if (res.ok) applyTelegramSettings(await res.json());
    } catch (_) {}
  }

  async function postTelegram(path, body) {
    const res = await fetch(`${API_BASE}/api/settings/telegram${path}`, {
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
        await fetch(`${API_BASE}/api/system/shutdown`, { method: "POST" });
        alert("Hệ thống đã dừng an toàn! Bạn có thể đóng cửa sổ trình duyệt này.");
        window.close();
      } catch (_) {
        alert("Đã gửi lệnh tắt hệ thống.");
      }
    });
  }

  // 6. Kết nối các tương tác trên giao diện Compact mới
  const btnQuickSnap = document.getElementById("btn-quick-snapshot");
  if (btnQuickSnap) {
    btnQuickSnap.addEventListener("click", () => {
      const elBtnSnap = document.getElementById("btn-snapshot");
      if (elBtnSnap) elBtnSnap.click();
    });
  }

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


  // 10. Initialization Sequence
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

  console.log("[Dashboard] Initialized successfully. Connected to API:", API_BASE);
})();
