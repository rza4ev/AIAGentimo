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

# Check if VADER lexicon is available, download if not
try:
	nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
	nltk.download("vader_lexicon")

# Initialize sentiment analyzer for real-time sentiment analysis
sentiment_analyzer = SentimentIntensityAnalyzer()

# =====================================================
# 1) Environment configuration (API key and agent_id from .env)
# =====================================================

# Load environment variables from .env file
load_dotenv()

# Get API credentials from environment
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
AGENT_ID = os.getenv("ELEVEN_AGENT_ID")

# Build WebSocket server URL with agent ID
SERVER_URL = f"wss://api.elevenlabs.io/v1/convai/conversation?agent_id={AGENT_ID}"

# Default values for product substitution context
# Missing product üçün default, reason isə müştəri tərəfindən yazılacaq
DEFAULT_MISSING_PRODUCT = "Lactose-free milk 1L"
DEFAULT_REASON = ""

# =====================================================
# 2) Streamlit UI - MOBILE CALL INTERFACE
# =====================================================

# Configure Streamlit page settings for mobile-friendly call interface
st.set_page_config(page_title="Valio Aimo", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for mobile call interface styling
st.markdown("""
<style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .block-container {
        padding: 0.5rem !important;
        max-width: 100% !important;
    }
    .*{
	        display:flex;
			justify-content:center;
			align-items:center;
			text-align:center;
	}
    /* Call interface container */
    .call-container {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 25px;
        padding: 3rem 1.5rem 2.5rem 1.5rem;
        text-align: center;
        box-shadow: 0 15px 50px rgba(0,0,0,0.4);
        margin: 1rem 0 1.5rem 0;
        position: relative;
    }
    
    /* Caller name */
    .caller-name {
        color: white;
        font-size: 2.2rem;
		text-align:center;
        font-weight: 700;
        margin: 1.5rem 0 0.5rem 0;
        letter-spacing: -0.5px;
    }
    
    /* Call status */
    .call-status {
        color: rgba(255,255,255,0.85);
			text-align:center;
        font-size: 1.1rem;
        margin: 0.5rem 0 2.5rem 0;
        font-weight: 400;
    }
    
    /* Avatar circle */
    .avatar-circle {
        width: 130px;
        height: 130px;
        border-radius: 50%;
        background: linear-gradient(135deg, rgba(255,255,255,0.25), rgba(255,255,255,0.1));
        margin: 0 auto 1.5rem auto;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 4rem;
        border: 4px solid rgba(255,255,255,0.4);
        box-shadow: 0 8px 30px rgba(0,0,0,0.3);
    }
    
    /* Call button */
    .stButton button {
        background: linear-gradient(135deg, #00d26a 0%, #00b359 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 50% !important;
        width: 105px !important;
        height: 105px !important;
        font-size: 2.2rem !important;
        box-shadow: 0 8px 30px rgba(0,210,106,0.5) !important;
        margin: 0 auto !important;
        display: block !important;
        transition: all 0.3s ease !important;
        padding: 0 !important;
    }
    
    .stButton button:hover {
        transform: scale(1.08) !important;
        box-shadow: 0 10px 40px rgba(0,210,106,0.7) !important;
    }
    
    .stButton button:active {
        transform: scale(0.95) !important;
    }
    
    /* Calling animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.75; transform: scale(1.03); }
    }
    
    .calling-animation {
        animation: pulse 1.8s infinite ease-in-out;
    }
    
    /* Status indicator */
    .status-indicator {
        width: 12px;
        height: 12px;
        background: #00d26a;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
        box-shadow: 0 0 10px #00d26a;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.3; }
    }
    
    .status-indicator.active {
        animation: blink 1.5s infinite;
    }
    
    /* Chat container */
    .chat-container {
        background: #f8f9fa;
        border-radius: 20px;
        padding: 1.5rem 1rem;
        margin: 1.5rem 0;
        max-height: 400px;
        overflow-y: auto;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    [data-testid="stSidebar"] input, 
    [data-testid="stSidebar"] textarea {
        background: rgba(255,255,255,0.15) !important;
        border: 1px solid rgba(255,255,255,0.3) !important;
        border-radius: 12px !important;
        color: white !important;
        padding: 0.8rem !important;
    }
    
    [data-testid="stSidebar"] input::placeholder,
    [data-testid="stSidebar"] textarea::placeholder {
        color: rgba(255,255,255,0.6) !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        border-radius: 15px !important;
        color: white !important;
        font-weight: 600 !important;
        font-size: 1.1rem !important;
        padding: 1rem 1.5rem !important;
    }
    
    .streamlit-expanderContent {
        background: white;
        border-radius: 0 0 15px 15px;
        padding: 1.5rem !important;
        border: none !important;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 0.8rem 1.5rem;
        font-weight: 600;
        background: #f0f2f6;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white !important;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.7rem 2rem !important;
        font-weight: 600 !important;
        width: 100% !important;
    }
    
    /* Dataframe */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables for call log and call status
# Call log stores all user and agent messages with sentiment scores
if "call_log" not in st.session_state:
	st.session_state["call_log"] = []
if "is_calling" not in st.session_state:
	st.session_state["is_calling"] = False

# Initialize product substitution context in session state
if "missing_product" not in st.session_state:
	st.session_state["missing_product"] = DEFAULT_MISSING_PRODUCT
if "reason" not in st.session_state:
	st.session_state["reason"] = DEFAULT_REASON

# Main call interface - displays agent avatar, name, and call status
st.markdown('<div class="call-container">', unsafe_allow_html=True)
st.markdown('<div class="avatar-circle">🤖</div>', unsafe_allow_html=True)
st.markdown('<div class="caller-name">Valio Aimo</div>', unsafe_allow_html=True)

# Display call status with animated indicator during active call
if st.session_state["is_calling"]:
	st.markdown('<div class="call-status calling-animation"><span class="status-indicator active"></span>Call in progress...</div>', unsafe_allow_html=True)
else:
	st.markdown('<div class="call-status"><span class="status-indicator"></span>Ready to call</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Create empty containers for status updates and message log
status_box = st.empty()
log_box = st.empty()

# Sidebar - Dynamic context configuration for product substitution
with st.sidebar:
	st.title("⚙️ Settings")
	
	# Form for updating dynamic variables sent to the agent
	with st.form("dynamic_form"):
		missing_input = st.text_input(
			"Missing Product",
			value=st.session_state["missing_product"]
		)
		reason_input = st.text_area(
			"Reason / Customer note",
			value=st.session_state["reason"],
			height=100,
			help="Example: lactose-free only, low fat, similar taste, no nut traces, etc."
		)

		save_btn = st.form_submit_button("💾 Save Changes")

		# Update session state when settings are saved
		if save_btn:
			st.session_state["missing_product"] = missing_input
			st.session_state["reason"] = reason_input
			st.success("✅ Settings updated")

	# Display current configuration summary
	st.markdown("---")
	st.markdown("**Current Configuration:**")
	st.caption(f"🔸 Missing product: {st.session_state['missing_product']}")
	st.caption(f"🔸 Customer reason: {st.session_state['reason'] or '—'}")

# =====================================================
# 3) Audio configuration
# =====================================================

# Audio stream parameters for microphone input and speaker output
CHUNK = 1024  # Number of frames per buffer
AUDIO_RATE = 16000  # Sample rate in Hz (16kHz for voice)
FORMAT = pyaudio.paInt16  # 16-bit audio format
CHANNELS = 1  # Mono audio

# Initialize PyAudio interface
audio_interface = pyaudio.PyAudio()

# =====================================================
# 4) Call log and sentiment helper functions
# =====================================================

def compute_sentiment(text: str) -> float:
	"""
	Compute sentiment score for given text using VADER.
	
	Args:
		text: Input text to analyze
		
	Returns:
		Compound sentiment score between -1 (negative) and 1 (positive)
	"""
	if not text:
		return 0.0
	scores = sentiment_analyzer.polarity_scores(text)
	return scores["compound"]


def log_message(speaker: str, text: str):
	"""
	Log a message to the call log with sentiment analysis.
	
	Args:
		speaker: Either "user" or "agent"
		text: The message text
	"""
	if not text:
		return

	# Calculate sentiment score for the message
	sent = compute_sentiment(text)

	# Add message to call log with metadata
	st.session_state["call_log"].append(
		{"speaker": speaker, "text": text, "sentiment": sent}
	)

	# Display all messages in the log box with sentiment indicators
	with log_box.container():
		for msg in st.session_state["call_log"]:
			# Determine chat message role for styling
			role = "assistant" if msg["speaker"] == "agent" else "user"

			# Classify sentiment into positive/neutral/negative categories
			if msg["sentiment"] >= 0.05:
				sent_label = "😊 positive"
				sent_color = "#00d26a"
			elif msg["sentiment"] <= -0.05:
				sent_label = "😟 negative"
				sent_color = "#ff4757"
			else:
				sent_label = "😐 neutral"
				sent_color = "#95a5a6"

			# Display message with sentiment indicator
			with st.chat_message(role):
				st.write(msg["text"])
				st.markdown(
					f'<span style="color: {sent_color}; font-size: 0.85rem; font-weight: 600;">{sent_label} ({msg["sentiment"]:.2f})</span>',
					unsafe_allow_html=True
				)

# =====================================================
# 5) Bidirectional audio helper functions
# =====================================================

async def send_mic_audio(ws, in_stream):
	"""
	Continuously capture audio from microphone and send to WebSocket.
	
	Args:
		ws: WebSocket connection
		in_stream: PyAudio input stream (microphone)
	"""
	try:
		while True:
			# Read audio chunk from microphone
			audio_data = in_stream.read(CHUNK, exception_on_overflow=False)
			# Encode audio as base64 for transmission
			audio_b64 = base64.b64encode(audio_data).decode("utf-8")
			# Send audio chunk to server
			await ws.send(json.dumps({"user_audio_chunk": audio_b64}))
			# Small delay to prevent overwhelming the connection
			await asyncio.sleep(0.01)
	except (ConnectionClosedOK, ConnectionClosedError, asyncio.CancelledError):
		# Connection closed gracefully or task cancelled
		return


async def receive_agent_events(ws, out_stream):
	"""
	Receive and process events from the agent via WebSocket.
	Handles transcripts, agent responses, audio playback, and ping/pong.
	
	Args:
		ws: WebSocket connection
		out_stream: PyAudio output stream (speakers)
	"""
	try:
		while True:
			# Receive message from WebSocket
			msg = await ws.recv()

			# Parse JSON message
			try:
				data = json.loads(msg)
			except json.JSONDecodeError:
				continue

			event_type = data.get("type")

			# Handle user transcript events (speech-to-text of user's voice)
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

			# Handle agent response events (agent's text responses)
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

			# Handle audio events (agent's voice output)
			if event_type == "audio":
				evt = data.get("audio_event", data)
				audio_b64 = evt.get("audio_base_64") or evt.get("audio")
				if audio_b64:
					# Decode base64 audio and play through speakers
					audio_chunk = base64.b64decode(audio_b64)
					out_stream.write(audio_chunk)

			# Handle ping events (keep-alive mechanism)
			if event_type == "ping":
				ping_evt = data.get("ping_event", {})
				event_id = ping_evt.get("event_id")
				if event_id is not None:
					# Respond with pong to keep connection alive
					pong = {"type": "pong", "event_id": event_id}
					await ws.send(json.dumps(pong))
	except (ConnectionClosedOK, ConnectionClosedError, asyncio.CancelledError):
		# Connection closed gracefully or task cancelled
		return

# =====================================================
# 6) Main async loop (bidirectional audio)
# =====================================================

async def audio_loop():
	"""
	Main async function that manages the entire call session.
	Establishes WebSocket connection, sets up audio streams,
	sends dynamic context, and coordinates bidirectional audio.
	"""
	status_box.info("🔌 Connecting to server...")

	# Get current product substitution context from session state
	missing = st.session_state["missing_product"]
	reason = st.session_state["reason"]

	in_stream = None
	out_stream = None

	try:
		# Establish WebSocket connection to ElevenLabs Conversational AI
		async with websockets.connect(SERVER_URL) as ws:
			st.session_state["is_calling"] = True
			status_box.success("✅ Call connected")
			log_message("agent", "🔗 Connection established")

			# Open audio output stream for agent's voice
			out_stream = audio_interface.open(
				format=FORMAT,
				channels=CHANNELS,
				rate=AUDIO_RATE,
				output=True,
				frames_per_buffer=CHUNK,
			)

			# Open audio input stream for user's microphone
			in_stream = audio_interface.open(
				format=FORMAT,
				channels=CHANNELS,
				rate=AUDIO_RATE,
				input=True,
				frames_per_buffer=CHUNK,
			)

			# Send dynamic variables to agent (product substitution context)
			await ws.send(
				json.dumps(
					{
						"type": "conversation_initiation_client_data",
						"dynamic_variables": {
							"missing_product": missing,
							"reason": reason,
						},
					}
				)
			)

			# Send initial user message to start the conversation
			user_init_text = "Hi, I'm ready to review my order."
			await ws.send(
				json.dumps(
					{
						"type": "user_message",
						"text": user_init_text,
					}
				)
			)
			log_message("user", user_init_text)

			# Create concurrent tasks for sending and receiving audio
			send_task = asyncio.create_task(send_mic_audio(ws, in_stream))
			recv_task = asyncio.create_task(receive_agent_events(ws, out_stream))

			# Run both tasks concurrently until completion or error
			try:
				await asyncio.gather(send_task, recv_task)
			except (ConnectionClosedOK, ConnectionClosedError):
				st.session_state["is_calling"] = False
				status_box.info("📞 Call ended")
	except Exception as e:
		# Handle any connection errors
		st.session_state["is_calling"] = False
		status_box.error(f"❌ Connection error: {e}")
	finally:
		# Clean up audio streams
		if in_stream is not None:
			in_stream.stop_stream()
			in_stream.close()
		if out_stream is not None:
			out_stream.stop_stream()
			out_stream.close()

# =====================================================
# 7) Streamlit button
# =====================================================

def start_agent():
	"""
	Start the agent call session by running the async audio loop.
	"""
	asyncio.run(audio_loop())


# Call button - initiates or displays active call status
if st.button("📞" if not st.session_state["is_calling"] else "📞"):
	start_agent()

# =====================================================
# 8) Call log & NLTK Sentiment dashboard (WITH CHARTS)
# =====================================================

# Display analytics only if there are logged messages
if st.session_state.get("call_log"):
	st.markdown("---")
	
	# Expandable section for call statistics and analytics
	with st.expander("📊 Call Statistics & Analytics", expanded=False):
		# Convert call log to DataFrame for analysis
		df = pd.DataFrame(st.session_state["call_log"])

		def sentiment_label(score: float) -> str:
			"""
			Classify sentiment score into categorical label.
			
			Args:
				score: Compound sentiment score
				
			Returns:
				String label: "positive", "neutral", or "negative"
			"""
			if score >= 0.05:
				return "positive"
			elif score <= -0.05:
				return "negative"
			else:
				return "neutral"

		# Add sentiment label column for easier filtering and grouping
		df["sentiment_label"] = df["sentiment"].apply(sentiment_label)

		# Filter panel for customizing data view
		st.markdown("### 🔎 Filter Options")
		col_f1, col_f2 = st.columns(2)
		
		with col_f1:
			# Filter by speaker (user or agent)
			speaker_filter = st.multiselect(
				"Speaker",
				options=["user", "agent"],
				default=["user", "agent"],
			)
		
		with col_f2:
			# Filter by sentiment type
			sentiment_types = ["positive", "neutral", "negative"]
			sentiment_filter = st.multiselect(
				"Sentiment Type",
				options=sentiment_types,
				default=sentiment_types,
			)

		# Filter by sentiment score range
		sent_min, sent_max = st.slider(
			"Sentiment Score Range",
			min_value=-1.0,
			max_value=1.0,
			value=(-1.0, 1.0),
			step=0.05,
		)

		# Apply all filters to the DataFrame
		filtered_df = df[
			df["speaker"].isin(speaker_filter)
			& df["sentiment_label"].isin(sentiment_filter)
			& df["sentiment"].between(sent_min, sent_max)
		]

		# Show warning if no data matches filters
		if filtered_df.empty:
			st.warning("⚠️ No messages match the current filters. Try adjusting them.")
		else:
			# Create tabs for different analytics views
			tab_overview, tab_trend, tab_details = st.tabs(["📌 Overview", "📈 Trend Analysis", "📋 Message Details"])

			# =======================
			# 8.1 Overview tab - Key metrics and distribution
			# =======================
			with tab_overview:
				st.markdown("### Key Metrics")

				# Separate agent and user messages for individual metrics
				agent_df = filtered_df[filtered_df["speaker"] == "agent"]
				user_df = filtered_df[filtered_df["speaker"] == "user"]

				# Display key metrics in columns
				col1, col2, col3 = st.columns(3)
				with col1:
					st.metric("💬 Total Messages", len(filtered_df))
				with col2:
					agent_avg = agent_df["sentiment"].mean() if not agent_df.empty else 0
					st.metric("🤖 Agent Avg Sentiment", f"{agent_avg:.3f}")
				with col3:
					user_avg = user_df["sentiment"].mean() if not user_df.empty else 0
					st.metric("👤 User Avg Sentiment", f"{user_avg:.3f}")

				st.markdown("---")
				st.markdown("### 📊 Sentiment Distribution")

				# Group messages by speaker and sentiment label
				counts = (
					filtered_df
					.groupby(["speaker", "sentiment_label"])
					.size()
					.reset_index(name="count")
				)

				# Create bar chart if data exists
				if not counts.empty:
					pivot = counts.pivot(
						index="sentiment_label",
						columns="speaker",
						values="count"
					).fillna(0)

					st.bar_chart(pivot, use_container_width=True)
				else:
					st.info("Not enough data for visualization.")

			# =======================
			# 8.2 Trend tab - Sentiment over time
			# =======================
			with tab_trend:
				st.markdown("### Sentiment Trends Over Time")

				# Reset index for chronological ordering
				trend_df = filtered_df.reset_index(drop=True)

				# Separate user and agent trends
				user_trend = trend_df[trend_df["speaker"] == "user"][["sentiment"]].reset_index(drop=True)
				agent_trend = trend_df[trend_df["speaker"] == "agent"][["sentiment"]].reset_index(drop=True)

				col_t1, col_t2 = st.columns(2)
				
				with col_t1:
					st.markdown("#### 👤 User Sentiment Trend")
					if not user_trend.empty:
						st.line_chart(user_trend, use_container_width=True)
					else:
						st.caption("No user messages in filtered data.")

				with col_t2:
					st.markdown("#### 🤖 Agent Sentiment Trend")
					if not agent_trend.empty:
						st.line_chart(agent_trend, use_container_width=True)
					else:
						st.caption("No agent messages in filtered data.")

			# =======================
			# 8.3 Details tab - Full message log and export
			# =======================
			with tab_details:
				st.markdown("### Message Details")

				# Display full filtered message log as table
				st.dataframe(
					filtered_df[["speaker", "text", "sentiment", "sentiment_label"]],
					use_container_width=True,
					hide_index=True
				)

				# Excel export functionality
				def make_excel_download(dataframe):
					"""
					Create Excel file from DataFrame for download.
					
					Args:
						dataframe: pandas DataFrame to export
						
					Returns:
						BytesIO buffer containing Excel file
					"""
					output = BytesIO()
					dataframe.to_excel(output, index=False)
					output.seek(0)
					return output

				# Generate Excel file and provide download button
				excel_file = make_excel_download(filtered_df)
				st.download_button(
					"⬇️ Download Call Log (Excel)",
					data=excel_file,
					file_name="call_log_filtered.xlsx",
					mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
				)
else:
	# Show info message when no call data exists yet
	st.info("📞 Start a call session to see statistics and analytics.")
