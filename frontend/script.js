/**
 * 前端：登录态 + 产品态对话页。
 * - 工号密码登录，Bearer token 存 localStorage
 * - 页头展示姓名 / 工号 / 身份；聊天不再传 role（由服务端 token 决定）
 * - 助手消息按 Markdown 渲染；不向员工展示路由 / 检索 / 工具等调试标签
 */

const loginView = document.getElementById("login-view");
const appView = document.getElementById("app-view");
const loginForm = document.getElementById("login-form");
const loginError = document.getElementById("login-error");
const loginBtn = document.getElementById("login-btn");
const employeeIdEl = document.getElementById("employee-id");
const passwordEl = document.getElementById("password");
const userDisplay = document.getElementById("user-display");
const logoutBtn = document.getElementById("logout-btn");

const messagesEl = document.getElementById("messages");
const inputEl = document.getElementById("input");
const sendBtn = document.getElementById("send");

const SESSION_KEY = "eka_session_id";
const TOKEN_KEY = "eka_auth_token";

const ROLE_LABELS = {
  intern: "实习生",
  employee: "员工",
  admin: "管理员",
};

if (typeof marked !== "undefined") {
  marked.setOptions({
    breaks: true,
    gfm: true,
  });
}

/**
 * @returns {string|null}
 */
function getToken() {
  try {
    return localStorage.getItem(TOKEN_KEY) || null;
  } catch (_) {
    return null;
  }
}

/**
 * @param {string} token
 */
function saveToken(token) {
  try {
    localStorage.setItem(TOKEN_KEY, token);
  } catch (_) {
    // 隐私模式等场景忽略
  }
}

function clearToken() {
  try {
    localStorage.removeItem(TOKEN_KEY);
  } catch (_) {
    // ignore
  }
}

function getSessionId() {
  try {
    return localStorage.getItem(SESSION_KEY) || null;
  } catch (_) {
    return null;
  }
}

function saveSessionId(sessionId) {
  if (!sessionId) {
    return;
  }
  try {
    localStorage.setItem(SESSION_KEY, sessionId);
  } catch (_) {
    // ignore
  }
}

function clearSessionId() {
  try {
    localStorage.removeItem(SESSION_KEY);
  } catch (_) {
    // ignore
  }
}

/**
 * 将助手回复的 Markdown 转为安全 HTML。
 * @param {string} text
 * @returns {string}
 */
function renderMarkdown(text) {
  const raw = text || "";
  if (typeof marked === "undefined") {
    const div = document.createElement("div");
    div.textContent = raw;
    return div.innerHTML.replace(/\n/g, "<br>");
  }
  const html = marked.parse(raw);
  if (typeof DOMPurify !== "undefined") {
    return DOMPurify.sanitize(html);
  }
  return html;
}

/**
 * @param {"user"|"assistant"} role
 * @param {string} text
 */
function appendBubble(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `bubble ${role}`;

  if (role === "assistant") {
    bubble.classList.add("md");
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = text;
  }

  messagesEl.appendChild(bubble);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function setLoading(loading) {
  sendBtn.disabled = loading;
  inputEl.disabled = loading;
  sendBtn.textContent = loading ? "思考中…" : "发送";
}

/**
 * @param {{ name: string, employee_id: string, role: string }} user
 */
function renderUserBar(user) {
  const roleLabel = ROLE_LABELS[user.role] || user.role;
  userDisplay.textContent = `${user.name} · ${user.employee_id} · ${roleLabel}`;
}

function showLogin(errorMsg) {
  appView.classList.add("hidden");
  loginView.classList.remove("hidden");
  if (errorMsg) {
    loginError.hidden = false;
    loginError.textContent = errorMsg;
  } else {
    loginError.hidden = true;
    loginError.textContent = "";
  }
}

function showApp(user) {
  loginView.classList.add("hidden");
  appView.classList.remove("hidden");
  renderUserBar(user);
  if (!messagesEl.querySelector(".hint") && messagesEl.childElementCount === 0) {
    const welcome = document.createElement("div");
    welcome.className = "hint";
    welcome.textContent =
      "你好，我是企业知识助手。可以问我制度规定、年假天数或报销相关问题。";
    messagesEl.appendChild(welcome);
  }
  inputEl.focus();
}

function logout(message) {
  clearToken();
  clearSessionId();
  messagesEl.innerHTML = "";
  passwordEl.value = "";
  showLogin(message || "");
}

/**
 * 带 Bearer 的 fetch；401 时清登录态并回到登录页。
 * @param {string} url
 * @param {RequestInit} [options]
 */
async function authFetch(url, options) {
  const token = getToken();
  const headers = new Headers((options && options.headers) || {});
  if (token) {
    headers.set("Authorization", `Bearer ${token}`);
  }
  if (options && options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const resp = await fetch(url, { ...options, headers });
  if (resp.status === 401) {
    logout("登录已失效，请重新登录");
    throw new Error("UNAUTHORIZED");
  }
  return resp;
}

/**
 * 用已有 token 拉取 /api/auth/me，恢复页头。
 * @returns {Promise<boolean>} 是否仍处于登录态
 */
async function restoreSession() {
  const token = getToken();
  if (!token) {
    showLogin();
    return false;
  }
  try {
    const resp = await authFetch("/api/auth/me");
    if (!resp.ok) {
      logout("登录已失效，请重新登录");
      return false;
    }
    const user = await resp.json();
    showApp(user);
    return true;
  } catch (err) {
    if (err && err.message === "UNAUTHORIZED") {
      return false;
    }
    showLogin("无法验证登录状态，请重新登录");
    clearToken();
    return false;
  }
}

async function handleLogin(event) {
  event.preventDefault();
  const employeeId = employeeIdEl.value.trim();
  const password = passwordEl.value;
  if (!employeeId || !password) {
    return;
  }

  loginBtn.disabled = true;
  loginError.hidden = true;
  loginError.textContent = "";

  try {
    const resp = await fetch("/api/auth/login", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ employee_id: employeeId, password }),
    });
    if (!resp.ok) {
      let detail = `登录失败（HTTP ${resp.status}）`;
      try {
        const err = await resp.json();
        if (err.detail) {
          detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
        }
      } catch (_) {
        // ignore
      }
      loginError.hidden = false;
      loginError.textContent = detail;
      return;
    }
    const data = await resp.json();
    saveToken(data.token);
    // 换账号登录时清掉旧会话，避免串会话画像
    clearSessionId();
    showApp({
      name: data.name,
      employee_id: data.employee_id,
      role: data.role,
    });
  } catch (err) {
    loginError.hidden = false;
    loginError.textContent = `网络异常：${err.message || err}`;
  } finally {
    loginBtn.disabled = false;
  }
}

async function sendMessage() {
  const message = inputEl.value.trim();
  if (!message) {
    return;
  }
  if (!getToken()) {
    logout("请先登录");
    return;
  }
  // 未登录时不应停留在聊天页（防御：异常状态或旧缓存残留）
  if (appView.classList.contains("hidden")) {
    showLogin("请先登录后再发送消息");
    return;
  }

  const hint = messagesEl.querySelector(".hint");
  if (hint) {
    hint.remove();
  }

  appendBubble("user", message);
  inputEl.value = "";
  setLoading(true);

  const body = { message };
  const sid = getSessionId();
  if (sid) {
    body.session_id = sid;
  }

  try {
    const resp = await authFetch("/api/chat", {
      method: "POST",
      body: JSON.stringify(body),
    });

    if (!resp.ok) {
      let detail = `请求失败（HTTP ${resp.status}）`;
      try {
        const err = await resp.json();
        if (err.detail) {
          detail = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
        }
      } catch (_) {
        // ignore
      }
      const errBubble = document.createElement("div");
      errBubble.className = "bubble assistant error";
      errBubble.textContent = detail;
      messagesEl.appendChild(errBubble);
      return;
    }

    const data = await resp.json();
    if (data.session_id) {
      saveSessionId(data.session_id);
    }
    appendBubble("assistant", data.answer || "（空回答）");
  } catch (err) {
    if (err && err.message === "UNAUTHORIZED") {
      return;
    }
    const errBubble = document.createElement("div");
    errBubble.className = "bubble assistant error";
    errBubble.textContent = `网络异常：${err.message || err}`;
    messagesEl.appendChild(errBubble);
  } finally {
    setLoading(false);
    if (!appView.classList.contains("hidden")) {
      inputEl.focus();
    }
  }
}

logoutBtn.addEventListener("click", () => logout());
sendBtn.addEventListener("click", sendMessage);
inputEl.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendMessage();
  }
});

loginForm.addEventListener("submit", handleLogin);

// 启动：默认已是登录页；有有效 token 再切到聊天
restoreSession();
