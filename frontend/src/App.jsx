import { useState, useRef, useEffect } from "react";
import ReactMarkdown from "react-markdown";
import "./App.css";

const API_URL = "http://localhost:8000/api";

// Fix 5 — suggestion chips for first-time users
const SUGGESTIONS = [
  "🏔️ I want adventure sports and hiking in the mountains",
  "🏖️ Looking for a relaxing beach vacation with spa treatments",
  "🎭 Recommend cultural cities with museums and history",
  "💰 Best budget destinations in Southeast Asia",
  "💎 Luxury resort experience with private beach",
  "👨‍👩‍👧 Family-friendly destination with activities for kids",
];

// Fix 2 — unique ID generator instead of array index
let messageId = 0;
const nextId = () => ++messageId;

function Message({ role, content, metadata }) {
  return (
    <div className={`message ${role}`}>
      <div className="message-bubble">
        {/* Fix 1 — render markdown instead of raw text */}
        <ReactMarkdown>{content}</ReactMarkdown>
        {metadata && (
          <div className="metadata">
            <span className="style-badge">{metadata.predicted_style}</span>
            <span className="confidence">
              {(metadata.style_confidence * 100).toFixed(0)}% confidence
            </span>
            <span className="chunks">
              {metadata.chunks_retrieved} sources
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function TypingIndicator() {
  return (
    <div className="message assistant">
      <div className="message-bubble typing">
        <span></span>
        <span></span>
        <span></span>
      </div>
    </div>
  );
}

// Fix 5 — suggestion chips component
function Suggestions({ onSelect }) {
  return (
    <div className="suggestions">
      <p className="suggestions-label">Try asking:</p>
      <div className="suggestions-grid">
        {SUGGESTIONS.map((s, i) => (
          <button
            key={i}
            className="suggestion-chip"
            onClick={() => onSelect(s)}
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: nextId(),
      role: "assistant",
      content:
        "Hi! I'm your **Smart Travel Planner**. Tell me what kind of trip you're looking for — adventure, relaxation, culture, budget, luxury, or family travel?",
      metadata: null,
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  const sendMessage = async (text) => {
    const messageText = text || input;
    if (!messageText.trim() || loading) return;

    // Hide suggestions after first message
    setShowSuggestions(false);

    // Fix 2 — use unique ID not array index
    const userMessage = {
      id: nextId(),
      role: "user",
      content: messageText,
      metadata: null,
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: messageText }),
      });

      if (!response.ok) {
        throw new Error(`API error: ${response.status}`);
      }

      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        {
          id: nextId(),
          role: "assistant",
          content: data.response,
          metadata: {
            predicted_style: data.predicted_style,
            style_confidence: data.style_confidence,
            chunks_retrieved: data.chunks_retrieved,
          },
        },
      ]);
    } catch (err) {
      setError("Failed to get a response. Is the backend running?");
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="app">
      <header className="header">
        <h1>✈️ Smart Travel Planner</h1>
        <p>AI-powered destination recommendations</p>
      </header>

      <main className="chat-window">
        {messages.map((msg) => (
          <Message
            key={msg.id}
            role={msg.role}
            content={msg.content}
            metadata={msg.metadata}
          />
        ))}

        {/* Fix 5 — show suggestions until first user message */}
        {showSuggestions && !loading && (
          <Suggestions onSelect={(s) => sendMessage(s)} />
        )}

        {loading && <TypingIndicator />}
        {error && <div className="error">{error}</div>}
        <div ref={bottomRef} />
      </main>

      <footer className="input-area">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe your dream trip... (Enter to send)"
          rows={2}
          disabled={loading}
        />
        <button onClick={() => sendMessage()} disabled={loading || !input.trim()}>
          {loading ? "..." : "Send"}
        </button>
      </footer>
    </div>
  );
}