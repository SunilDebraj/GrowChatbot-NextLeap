/* P5 client. Vanilla JS, no build step, no framework - the page is a demo of a
   facts product, and a bundler would be one more thing between a reviewer and
   the code that draws the answer.
 *
 * Two rules govern everything here:
 *
 * 1. Text goes in with textContent, never innerHTML. The answer is model-adjacent
 *    text, so treating it as markup is the one way this page could execute
 *    something. The page's CSP also forbids inline script, so this is the
 *    second of two independent barriers.
 * 2. The citation href comes from the API's structured citation.url field, not
 *    from parsing a markdown link out of the answer. The validator already
 *    guarantees exactly one URL in the answer and that it is the one the
 *    pipeline attached from chunk metadata, but a structured field is still the
 *    right source: parsing is where a crafted answer would get to choose its own
 *    href. The markdown syntax is stripped from the displayed text instead.
 */

(function () {
  "use strict";

  var thread = document.getElementById("thread");
  var form = document.getElementById("ask-form");
  var input = document.getElementById("question");
  var button = document.getElementById("ask-button");

  var MD_LINK = /\[([^\]]*)\]\(([^)]*)\)/g;

  /* Conversation memory. The last N questions live in this variable only - no
   * browser storage, no cookie - so a reload forgets them. They are sent with
   * each question so the server can carry a scheme into a follow-up ("and the
   * exit load?"). A question the server flagged as PII is never kept. */
  var MEMORY_TURNS = parseInt(form.getAttribute("data-memory-turns") || "0", 10) || 0;
  var history = [];

  function remember(question, payload) {
    if (!MEMORY_TURNS || (payload && payload["class"] === "pii")) {
      return;
    }
    history.push(question);
    if (history.length > MEMORY_TURNS) {
      history = history.slice(history.length - MEMORY_TURNS);
    }
  }

  /* The thread grows with the page, so each new message is scrolled into view
   * rather than scrolling an inner box. */
  function reveal(el) {
    if (el && el.scrollIntoView) {
      el.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  }

  /* The input grows with its text up to the CSS max-height. */
  function fit() {
    input.style.height = "auto";
    input.style.height = input.scrollHeight + "px";
  }

  function addUser(text) {
    var el = document.createElement("p");
    el.className = "bubble user";
    el.textContent = text;
    thread.appendChild(el);
    reveal(thread.lastElementChild);
  }

  function addPending() {
    var el = document.createElement("p");
    el.className = "bubble pending";
    el.textContent = "Looking in the sources...";
    thread.appendChild(el);
    reveal(thread.lastElementChild);
    return el;
  }

  /* The refusal footer is shown on every refusal (P5-T7). The refusal text
   * already ends with the facts-only line, so this is a visible marker rather
   * than new copy - it says the message is a refusal, not an answer. */
  function addAnswer(payload) {
    var wrap = document.createElement("div");
    wrap.className = "bubble answer";
    if (payload.route !== "answer") {
      wrap.classList.add("refusal");
    }

    var text = String(payload.answer || "");
    var body = document.createElement("p");
    body.className = "body";
    // Drop the markdown link the pipeline attached; the real link is rendered
    // below from citation.url, so the user sees it as a link and not as syntax.
    body.textContent = text
      .replace(MD_LINK, function (_m, label) {
        return label;
      })
      .trim();
    wrap.appendChild(body);

    var url = payload.citation && payload.citation.url;
    if (url) {
      var a = document.createElement("a");
      a.className = "citation";
      a.href = url;
      a.rel = "noopener noreferrer nofollow";
      a.target = "_blank";
      // A refusal's link is educational, not the source of an answer.
      a.textContent = (payload.route === "answer" ? "Source: " : "Learn more: ") +
        (payload.citation.label || "official page");
      wrap.appendChild(a);
    }

    if (payload.last_updated) {
      var stamp = document.createElement("p");
      stamp.className = "stamp";
      stamp.textContent = "Last updated from sources: " + payload.last_updated;
      wrap.appendChild(stamp);
    }

    if (payload.route !== "answer") {
      var footer = document.createElement("p");
      footer.className = "refusal-footer";
      footer.textContent = "Facts-only. No investment advice.";
      wrap.appendChild(footer);
    }

    thread.appendChild(wrap);
    reveal(thread.lastElementChild);
  }

  function addNotice(message) {
    var el = document.createElement("p");
    el.className = "bubble notice";
    el.textContent = message;
    thread.appendChild(el);
    reveal(thread.lastElementChild);
  }

  function ask(question) {
    addUser(question);
    var pending = addPending();
    button.disabled = true;

    fetch("/api/ask", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ question: question, history: history.slice() }),
    })
      .then(function (response) {
        return response.json().then(function (data) {
          return { ok: response.ok, data: data };
        });
      })
      .then(function (result) {
        pending.remove();
        if (result.ok) {
          addAnswer(result.data);
          remember(question, result.data);
        } else {
          // The API already sent a safe, human sentence. Show it as-is; never
          // invent error copy here that could disagree with it.
          addNotice((result.data && result.data.error && result.data.error.message) ||
            "The assistant cannot answer right now.");
        }
      })
      .catch(function () {
        pending.remove();
        addNotice("Could not reach the assistant. Check that the service is running.");
      })
      .then(function () {
        button.disabled = false;
        input.focus();
      });
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var question = input.value.trim();
    if (!question) {
      return;
    }
    input.value = "";
    fit();
    ask(question);
  });

  /* Enter sends, Shift+Enter starts a new line. */
  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (!button.disabled) {
        form.requestSubmit ? form.requestSubmit() : button.click();
      }
    }
  });
  input.addEventListener("input", fit);

  /* P5 pitfall: the examples are fixed in config. Clicking one fills the input
   * rather than sending immediately, so the demo can be narrated before the
   * answer appears. */
  var examples = document.querySelectorAll(".example");
  Array.prototype.forEach.call(examples, function (el) {
    el.addEventListener("click", function () {
      input.value = el.getAttribute("data-question") || "";
      fit();
      input.focus();
    });
  });
})();
