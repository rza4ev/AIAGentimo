Valio Aimo - AI-Powered Customer Call Interface
A real-time conversational AI interface for handling customer calls with voice interaction, sentiment analysis, and comprehensive call analytics. Built with Streamlit and ElevenLabs Conversational AI.

🌟 Features
📞 Real-Time Voice Interaction

Bidirectional audio streaming (microphone ↔ speakers)
WebSocket-based communication with ElevenLabs AI agent
Low-latency voice recognition and synthesis
Mobile-friendly call interface with intuitive controls

😊 Sentiment Analysis

Real-time NLTK VADER sentiment analysis
Automatic sentiment scoring for each message
Visual sentiment indicators (positive/neutral/negative)
Color-coded sentiment display in chat

📊 Advanced Analytics Dashboard

Overview Tab: Key metrics and sentiment distribution

Total message count
Average sentiment scores (agent & user)
Bar chart visualization


Trend Analysis Tab: Sentiment progression over time

Separate user and agent sentiment trends
Line chart visualizations


Message Details Tab: Complete call log

Filterable message history
Excel export functionality



🔧 Dynamic Context Management

Configurable product substitution scenarios
Real-time context updates during calls
Sidebar settings panel for easy configuration

🎨 Modern UI/UX

Mobile-optimized call interface
Gradient backgrounds and smooth animations
Responsive design with custom CSS
Professional call screen aesthetic

🚀 Getting Started
Prerequisites

Python 3.8 or higher
ElevenLabs API account
Microphone and speakers/headphones
Internet connection

Installation

Clone the repository

bash   git clone https://github.com/yourusername/valio-aimo.git
   cd valio-aimo

Create a virtual environment

bash   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies

bash   pip install -r requirements.txt

Download NLTK data (automatic on first run, or manually)

python   python -c "import nltk; nltk.download('vader_lexicon')"
Configuration

Create a .env file in the project root:

env   ELEVEN_API_KEY=your_elevenlabs_api_key_here
   ELEVEN_AGENT_ID=your_agent_id_here

Get your ElevenLabs credentials:

Sign up at ElevenLabs
Create a Conversational AI agent
Copy your API key and Agent ID



Running the Application
bashstreamlit run app.py
The application will open in your default browser at http://localhost:8501
📖 Usage
Starting a Call

Configure Settings (optional):

Click the sidebar to expand settings
Enter missing product information
Enter substitute product details
Provide substitution reason
Click "💾 Save Changes"


Initiate Call:

Click the green phone button (📞)
Grant microphone permissions when prompted
Wait for connection establishment


During the Call:

Speak naturally into your microphone
Agent responses will play through your speakers
Real-time transcripts appear in the chat
Sentiment scores are calculated automatically


End the Call:

Close the browser tab or stop the Streamlit server



Viewing Analytics

Expand the Analytics Panel:

Scroll to "📊 Call Statistics & Analytics"
Click to expand the section


Apply Filters:

Select speakers (user/agent)
Choose sentiment types
Adjust sentiment score range


Explore Tabs:

Overview: View key metrics and distribution
Trend Analysis: See sentiment progression
Message Details: Browse full conversation log


Export Data:

Navigate to "Message Details" tab
Click "⬇️ Download Call Log (Excel)"
Open in Excel or Google Sheets



🛠️ Technology Stack
ComponentTechnologyFrontendStreamlitAudio ProcessingPyAudioAI PlatformElevenLabs Conversational AISentiment AnalysisNLTK VADERWebSocketwebsockets libraryData ProcessingpandasEnvironmentpython-dotenv
🔑 Environment Variables
VariableDescriptionRequiredELEVEN_API_KEYYour ElevenLabs API key✅ YesELEVEN_AGENT_IDYour conversational AI agent ID✅ Yes
🎯 Use Cases

Customer Service: Handle product substitution inquiries
Order Verification: Confirm order changes with customers
Sentiment Monitoring: Track customer satisfaction in real-time
Training & QA: Review call logs and agent performance
Analytics: Export data for further analysis

🐛 Troubleshooting
Audio Issues
No audio input/output:

Check microphone/speaker connections
Grant browser microphone permissions
Verify PyAudio installation: pip install --upgrade pyaudio
On Windows: May need to install Visual C++ Build Tools

Audio delays:

Check internet connection speed
Reduce CHUNK size in code (line 195)
Close other audio applications

Connection Issues
WebSocket connection failed:

Verify API credentials in .env
Check ElevenLabs account status
Ensure stable internet connection
Verify agent ID is correct

"Connection error" message:

Check if ElevenLabs API is operational
Verify API key hasn't expired
Review ElevenLabs service status page

Sentiment Analysis
VADER lexicon not found:
bashpython -c "import nltk; nltk.download('vader_lexicon')"
📊 Analytics & Metrics
Sentiment Scoring

Positive: Score ≥ 0.05 (😊 green)
Neutral: -0.05 < Score < 0.05 (😐 gray)
Negative: Score ≤ -0.05 (😟 red)

Exported Data Columns

speaker: "user" or "agent"
text: Message transcript
sentiment: Compound score (-1 to 1)
sentiment_label: "positive", "neutral", or "negative"

🔒 Security Considerations

Never commit .env file to version control
Keep API keys secure and rotate regularly
Use environment variables for all sensitive data
Review call logs before sharing externally
Monitor API usage to prevent unexpected charges

🤝 Contributing
Contributions are welcome! Please follow these steps:

Fork the repository
Create a feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

ElevenLabs for conversational AI platform
Streamlit for the web framework
NLTK for sentiment analysis tools
PyAudio for audio processing

📞 Support
For issues and questions:

Open an Issue
Check ElevenLabs Documentation
Review Streamlit Documentation

🗺️ Roadmap

 Multi-language support
 Custom sentiment models
 Call recording and playback
 Integration with CRM systems
 Advanced analytics dashboard
 Mobile app version
 Multi-agent support
 Real-time transcription export
