import os
import asyncio
import json
import base64

import streamlit as st
import websockets
import pyaudio
from dotenv import load_dotenv
import pandas as pd
from io import BytesIO

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

# =====================================================
# 0) NLTK VADER sentiment setup
# =====================================================

try:
	nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
	nltk.download("vader_lexicon")

sentiment_analyzer = SentimentIntensityAnalyzer()

# =====================================================
# 1) ENV konfiqurasiya (API key və agent_id .env-də)
# =====================================================

load_dotenv()

ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY", "CHANGE_ME")
AGENT_ID = os.getenv("ELEVEN_AGENT_ID", "agent_6301ka235a7cer6by0gpy8qx66j8")

SERVER_URL = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={AGENT_ID}"

# Default dəyərlər
DEFAULT_MISSING_PRODUCT = "Lactose-free milk 1L"
DEFAULT_SUBSTITUTE_PRODUCT = "Normal milk 1L"
DEFAULT_REASON = "It has similar taste and fat content, most customers accept this replacement."

# =====================================================
# 2) Streamlit UI – mobil call screen theming & layout
# =====================================================

st.set_page_config(
	page_title="Valio Aimo – AI Voice Agent",
	layout="centered",
	page_icon="📞",
)


def inject_custom_css():
	st.markdown(
		"""
		<style>
		/* Ana background – telefon zəngi havası, dark */
		.stApp {
			background: radial-gradient(circle at top, #111827 0%, #020617 50%, #000000 100%);
			color: #f9fafb;
		}

		/* Mobil üçün container-i ortala və max-width ver */
		.block-container {
			padding-top: 1.2rem;
			padding-bottom: 1.8rem;
			max-width: 480px;
		}

		/* Call ekranı wrapper */
		.call-wrapper {
			min-height: calc(100vh - 5rem);
			display: flex;
			flex-direction: column;
			align-items: center;
			justify-content: flex-start;
			text-align: center;
		}

		.call-brand {
			font-size: 0.9rem;
			font-weight: 500;
			color: #9ca3af;
			margin-top: 0.6rem;
			margin-bottom: 0.2rem;
		}

		.call-title {
			font-size: 1.5rem;
			font-weight: 700;
			margin-bottom: 0.1rem;
		}

		.call-subtitle {
			font-size: 0.9rem;
			color: #d1d5db;
			margin-bottom: 1.8rem;
			padding: 0 0.8rem;
		}

		/* Status text (Calling..., Ready və s.) */
		.call-status {
			font-size: 0.95rem;
			font-weight: 500;
			color: #a5b4fc;
			margin-bottom: 1.2rem;
			letter-spacing: 0.05em;
			text-transform: uppercase;
		}

		/* Zəng düyməsi üçün pulsating circle effekti */
		.call-circle-container {
			position: relative;
			width: 180px;
			height: 180px;
			display: flex;
			align-items: center;
			justify-content: center;
			margin-bottom: 1.3rem;
		}

		.call-circle-bg {
			position: absolute;
			width: 180px;
			height: 180px;
			border-radius: 50%;
			background: radial-gradient(circle, rgba(34,197,94,0.06), transparent 70%);
		}

		/* Dalğalanan halqalar – zəng olunur effekti */
		.ring-wave {
			position: absolute;
			border-radius: 50%;
			border: 2px solid rgba(52, 211, 153, 0.4);
			animation: ring 2.4s infinite;
		}
		.ring-wave.r1 {
			width: 120px;
			height: 120px;
			animation-delay: 0s;
		}
		.ring-wave.r2 {
			width: 150px;
			height: 150px;
			animation-delay: 0.35s;
		}
		.ring-wave.r3 {
			width: 180px;
			height: 180px;
			animation-delay: 0.7s;
		}

		@keyframes ring {
			0% {
				transform: scale(0.9);
				opacity: 0.85;
			}
			70% {
				transform: scale(1.06);
				opacity: 0;
			}
			100% {
				transform: scale(1.1);
				opacity: 0;
			}
		}

		/* Call düyməsi – tam dairəvi və böyük */
		.stButton > button {
			width: 110px;
			height: 110px;
			border-radius: 50%;
			border: none;
			font-size: 1.8rem;
			font-weight: 700;
			display: flex;
			align-items: center;
			justify-content: center;
			cursor: pointer;
			box-shadow: 0 16px 35px rgba(22, 163, 74, 0.6);
			background: radial-gradient(circle at 30% 20%, #bbf7d0 0%, #22c55e 45%, #15803d 100%);
			color: #022c22;
		}
		.stButton > button:hover {
			filter: brightness(1.05);
			transform: translateY(-1px);
			box-shadow: 0 18px 40px rgba(21, 128, 61, 0.8);
		}
		.stButton > button:active {
			transform: translateY(1px) scale(0.97);
			box-shadow: 0 10px 24px rgba(21, 128, 61, 0.7);
		}

		.call-hint {
			font-size: 0.8rem;
			color: #9ca3af;
			margin-top: 0.2rem;
		}

		.call-footer {
			font-size: 0.8rem;
			color: #6b7280;
			margin-top: 2rem;
		}

		/* Chat bubbles aşağıda (call log üçün) */
		[data-testid="stChatMessage"] {
			border-radius: 16px;
			padding: 0.55rem 0.75rem;
			margin-bottom: 0.35rem;
			box-shadow: 0 4px 10px rgba(15, 23, 42, 0.5);
			background-color: rgba(15,23,42,0.9);
			border: 1px solid rgba(55, 65, 81, 0.8);
		}
		[data-testid="stChatMessage"]:has(div[aria-label="assistant"]) {
			border-left: 3px solid #22c55e;
		}
		[data-testid="stChatMessage"]:has(div[aria-label="user"]) {
			border-left: 3px solid #38bdf8;
		}

		.section-title {
			font-size: 1rem;
			font-weight: 600;
			margin-top: 1.2rem;
			margin-bottom: 0.4rem;
			display: flex;
			align-items: center;
			gap: 0.35rem;
		}
		.section-title span.emoji {
			font-size: 1.1rem;
		}

		.analytics-card {
			border-radius: 14px;
			background-color: rgba(15,23,42,0.96);
			padding: 0.9rem 1rem;
			border: 1px solid rgba(55,65,81,0.9);
			box-shadow: 0 10px 24px rgba(0,0,0,0.7);
		}

		/* Sidebar – sadəcə dark, optional */
		[data-testid="stSidebar"] {
			background: #020617;
			color: #e5e7eb;
		}
		[data-testid="stSidebar"] * {
			color: #e5e7eb !important;
		}
		[data-testid="stSidebar"] input,
		[data-testid="stSidebar"] textarea {
			background-color: rgba(15,23,42,0.9) !important;
			border-radius: 10px !important;
			border: 1px solid rgba(156, 163, 175, 0.7) !important;
		}

		@media (max-width: 768px) {
			.call-title {
				font-size: 1.4rem;
			}
			.call-subtitle {
				font-size: 0.86rem;
			}
			.call-circle-container {
				width: 160px;
				height: 160px;
			}
			.call-circle-bg {
				width: 160px;
				height: 160px;
			}
		}
		</style>
		""",
		unsafe_allow_html=True,
	)


inject_custom_css()

# Sadə call_status (yalnız UI üçün)
if "call_status" not in st.session_state:
	st.session_state["call_status"] = "Ready to call"

# =====================================================
# 2.1 Sidebar – dynamic context (eyni qalsın)
# =====================================================

if "call_log" not in st.session_state:
	st.session_state["call_log"] = []

if "missing_product" not in st.session_state:
	st.session_state["missing_product"] = DEFAULT_MISSING_PRODUCT
if "substitute_product" not in st.session_state:
	st.session_state["substitute_product"] = DEFAULT_SUBSTITUTE_PRODUCT
if "reason" not in st.session_state:
	st.session_state["reason"] = DEFAULT_REASON

st.sidebar.header("⚙️ Dynamic call context")

with st.sidebar.form("dynamic_form"):
	missing_input = st.text_input(
		"Missing product",
		value=st.session_state["missing_product"]
	)
	substitute_input = st.text_input(
		"Substitute product",
		value=st.session_state["substitute_product"]
	)
	reason_input = st.text_area(
		"Reason for substitution",
		value=st.session_state["reason"],
		height=80
	)

	save_btn = st.form_submit_button("Save inputs")

	if save_btn:
		st.session_state["missing_product"] = missing_input
		st.session_state["substitute_product"] = substitute_input
		st.session_state["reason"] = reason_input
		st.success("Inputs updated ✅")

st.sidebar.write("---")
st.sidebar.write("📦 Current values:")
st.sidebar.write(f"• Missing: {st.session_state['missing_product']}")
st.sidebar.write(f"• Substitute: {st.session_state['substitute_product']}")
st.sidebar.write(f"• Reason: {st.session_state['reason']}")

# =====================================================
# 3) Audio konfiqurasiyası
# =====================================================

CHUNK = 1024
AUDIO_RATE = 16000
FORMAT = pyaudio.paInt16
CHANNELS = 1

audio_interface = pyaudio.PyAudio()

# =====================================================
# 4) Call log + sentiment helper-ləri
# =====================================================

status_box = st.empty()
log_box = st.empty()

def compute_sentiment(text: str) -> float:
	if not text:
		return 0.0
	scores = sentiment_analyzer.polarity_scores(text)
	return scores["compound"]


def log_message(speaker: str, text: str):
	if not text:
		return

	sent = compute_sentiment(text)

	st.session_state["call_log"].append(
		{"speaker": speaker, "text": text, "sentiment": sent}
	)

	with log_box.container():
		st.markdown(
			'<div class="section-title"><span class="emoji">💬</span>Live call transcript & mood</div>',
			unsafe_allow_html=True,
		)
		for msg in st.session_state["call_log"]:
			role = "assistant" if msg["speaker"] == "agent" else "user"

			if msg["sentiment"] >= 0.05:
				sent_label = "positive"
			elif msg["sentiment"] <= -0.05:
				sent_label = "negative"
			else:
				sent_label = "neutral"

			with st.chat_message(role):
				st.write(msg["text"])
				st.caption(f"Sentiment: {msg['sentiment']:.3f} ({sent_label})")

# =====================================================
# 5) Iki tərəfli audio üçün helper funksiyalar
# =====================================================

async def send_mic_audio(ws, in_stream):
	try:
		while True:
			audio_data = in_stream.read(CHUNK, exception_on_overflow=False)
			audio_b64 = base64.b64encode(audio_data).decode("utf-8")
			await ws.send(json.dumps({"user_audio_chunk": audio_b64}))
			await asyncio.sleep(0.01)
	except (ConnectionClosedOK, ConnectionClosedError, asyncio.CancelledError):
		return


async def receive_agent_events(ws, out_stream):
	try:
		while True:
			msg = await ws.recv()

			try:
				data = json.loads(msg)
			except json.JSONDecodeError:
				continue

			event_type = data.get("type")

			if event_type == "user_transcript":
				evt = data.get("user_transcription_event", data)
				text = (
					evt.get("user_transcript")
					or evt.get("text")
					or data.get("text")
					or ""
				)
				if text:
					log_message("user", text)

			if event_type == "agent_response":
				evt = data.get("agent_response_event", data)
				text = (
					evt.get("agent_response")
					or evt.get("text")
					or data.get("text")
					or ""
				)
				if text:
					log_message("agent", text)

			if event_type == "audio":
				evt = data.get("audio_event", data)
				audio_b64 = evt.get("audio_base_64") or evt.get("audio")
				if audio_b64:
					audio_chunk = base64.b64decode(audio_b64)
					out_stream.write(audio_chunk)

			if event_type == "ping":
				ping_evt = data.get("ping_event", {})
				event_id = ping_evt.get("event_id")
				if event_id is not None:
					pong = {"type": "pong", "event_id": event_id}
					await ws.send(json.dumps(pong))
	except (ConnectionClosedOK, ConnectionClosedError, asyncio.CancelledError):
		return

# =====================================================
# 6) Esas async loop (iki tərəfli)
# =====================================================

async def audio_loop():
	status_box.info("🔌 Connecting to ElevenLabs Realtime API...")

	missing = st.session_state["missing_product"]
	substitute = st.session_state["substitute_product"]
	reason = st.session_state["reason"]

	in_stream = None
	out_stream = None

	try:
		async with websockets.connect(SERVER_URL) as ws:
			status_box.success("✅ Connected! You can start talking now.")
			log_message("agent", "🔗 Connected to ElevenLabs")

			out_stream = audio_interface.open(
				format=FORMAT,
				channels=CHANNELS,
				rate=AUDIO_RATE,
				output=True,
				frames_per_buffer=CHUNK,
			)

			in_stream = audio_interface.open(
				format=FORMAT,
				channels=CHANNELS,
				rate=AUDIO_RATE,
				input=True,
				frames_per_buffer=CHUNK,
			)

			await ws.send(
				json.dumps(
					{
						"type": "conversation_initiation_client_data",
						"dynamic_variables": {
							"missing_product": missing,
							"substitute_product": substitute,
							"reason": reason,
						},
					}
				)
			)

			user_init_text = "Hi, I’m ready to review my order."
			await ws.send(
				json.dumps(
					{
						"type": "user_message",
						"text": user_init_text,
					}
				)
			)
			log_message("user", user_init_text)

			send_task = asyncio.create_task(send_mic_audio(ws, in_stream))
			recv_task = asyncio.create_task(receive_agent_events(ws, out_stream))

			try:
				await asyncio.gather(send_task, recv_task)
			except (ConnectionClosedOK, ConnectionClosedError):
				status_box.info("📞 Conversation ended. You can start a new session.")
	except Exception as e:
		status_box.error(f"🚨 Connection failed: {e}")
	finally:
		if in_stream is not None:
			in_stream.stop_stream()
			in_stream.close()
		if out_stream is not None:
			out_stream.stop_stream()
			out_stream.close()

# =====================================================
# 7) Call screen – mobil UI + düymə
# =====================================================

def start_agent():
	asyncio.run(audio_loop())


st.markdown(
	"""
	<div class="call-wrapper">
		<div class="call-brand">Valio Aimo · Delivery AI</div>
		<div class="call-title">Customer call</div>
		<div class="call-subtitle">
			Missing product? Let’s quickly confirm the substitute with the customer.
		</div>
	</div>
	""",
	unsafe_allow_html=True,
)

status_text = st.session_state.get("call_status", "Ready to call")

with st.container():
	st.markdown(
		f"""
		<div class="call-wrapper">
			<div class="call-status">{status_text}</div>
			<div class="call-circle-container">
				<div class="call-circle-bg"></div>
				<div class="ring-wave r1"></div>
				<div class="ring-wave r2"></div>
				<div class="ring-wave r3"></div>
				<div>
		""",
		unsafe_allow_html=True,
	)

	# Ortadakı yumru zəng düyməsi
	if st.button("📞", key="call_button"):
		st.session_state["call_status"] = "Calling..."
		start_agent()
		st.session_state["call_status"] = "Ready to call"

	st.markdown(
		"""
				</div>
			</div>
			<div class="call-hint">
				Tap to start the AI call. Speak into your microphone as if you talk to the customer.
			</div>
			<div class="call-footer">
				UI only – backend is already connected, you can refine behaviour in ElevenLabs agent prompt.
			</div>
		</div>
		""",
		unsafe_allow_html=True,
	)

# =====================================================
# 8) Call log & NLTK Sentiment dashboard (interactive)
# =====================================================

st.markdown("---")
st.markdown(
	'<div class="section-title"><span class="emoji">📊</span>Call log & NLTK Sentiment dashboard (real-time)</div>',
	unsafe_allow_html=True,
)

def sentiment_label(score: float) -> str:
	if score >= 0.05:
		return "positive"
	elif score <= -0.05:
		return "negative"
	else:
		return "neutral"

if st.session_state.get("call_log"):
	df = pd.DataFrame(st.session_state["call_log"])
	df["sentiment_label"] = df["sentiment"].apply(sentiment_label)

	with st.container():
		with st.expander("🔎 Filter options", expanded=True):
			speaker_filter = st.multiselect(
				"Speaker",
				options=["user", "agent"],
				default=["user", "agent"],
			)

			sentiment_types = ["positive", "neutral", "negative"]
			sentiment_filter = st.multiselect(
				"Sentiment type",
				options=sentiment_types,
				default=sentiment_types,
			)

			sent_min, sent_max = st.slider(
				"Sentiment interval (compound score)",
				min_value=-1.0,
				max_value=1.0,
				value=(-1.0, 1.0),
				step=0.05,
			)

	filtered_df = df[
		df["speaker"].isin(speaker_filter)
		& df["sentiment_label"].isin(sentiment_filter)
		& df["sentiment"].between(sent_min, sent_max)
	]

	if filtered_df.empty:
		st.warning("Filtr nəticəsində heç bir mesaj qalmadı. Əvvəlcə zəngi başlat, sonra bura qayıt 🙌")
	else:
		tab_overview, tab_trend, tab_details = st.tabs(["📌 Overview", "📈 Trend", "📋 Details"])

		with tab_overview:
			st.markdown('<div class="analytics-card">', unsafe_allow_html=True)

			st.markdown("**Ümumi göstəricilər**")

			agent_df = filtered_df[filtered_df["speaker"] == "agent"]
			user_df = filtered_df[filtered_df["speaker"] == "user"]

			col1, col2, col3 = st.columns(3)
			with col1:
				st.metric("Mesaj sayı (filtrlənmiş)", len(filtered_df))
			with col2:
				st.metric(
					"Agent orta sentiment",
					round(agent_df["sentiment"].mean(), 3) if not agent_df.empty else "–"
				)
			with col3:
				st.metric(
					"User orta sentiment",
					round(user_df["sentiment"].mean(), 3) if not user_df.empty else "–"
				)

			st.markdown("#### Sentiment paylanması")

			counts = (
				filtered_df
				.groupby(["speaker", "sentiment_label"])
				.size()
				.reset_index(name="count")
			)

			if not counts.empty:
				pivot = counts.pivot(
					index="sentiment_label",
					columns="speaker",
					values="count"
				).fillna(0)
				st.bar_chart(pivot)
			else:
				st.info("Qrafik üçün kifayət qədər data yoxdur.")

			st.markdown("</div>", unsafe_allow_html=True)

		with tab_trend:
			st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
			st.markdown("### Sentiment trendi (mesaj ardıcıllığı üzrə)")

			trend_df = filtered_df.reset_index(drop=True)

			user_trend = trend_df[trend_df["speaker"] == "user"][["sentiment"]]
			agent_trend = trend_df[trend_df["speaker"] == "agent"][["sentiment"]]

			st.write("User sentiment trendi:")
			if not user_trend.empty:
				st.line_chart(user_trend)
			else:
				st.caption("User mesajı yoxdur (filtrə görə).")

			st.write("Agent sentiment trendi:")
			if not agent_trend.empty:
				st.line_chart(agent_trend)
			else:
				st.caption("Agent mesajı yoxdur (filtrə görə).")

			st.markdown("</div>", unsafe_allow_html=True)

		with tab_details:
			st.markdown('<div class="analytics-card">', unsafe_allow_html=True)
			st.markdown("### Detallı mesaj cədvəli")

			st.dataframe(
				filtered_df[["speaker", "text", "sentiment", "sentiment_label"]],
				use_container_width=True,
			)

			def make_excel_download(dataframe):
				output = BytesIO()
				dataframe.to_excel(output, index=False)
				output.seek(0)
				return output

			excel_file = make_excel_download(filtered_df)
			st.download_button(
				"⬇️ Download filtered call log (Excel)",
				data=excel_file,
				file_name="call_log_filtered.xlsx",
				mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
			)

			st.markdown("</div>", unsafe_allow_html=True)
else:
	st.info("Hələ heç bir call log yoxdur. Əvvəlcə zəngi başlat, sonra burada görünəcək 📞")
