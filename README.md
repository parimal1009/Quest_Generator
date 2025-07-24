# Fantasy Quest Generator 🏰⚔️

A powerful AI-powered fantasy quest generator that creates immersive RPG quests using both fine-tuned local models and cloud-based AI. Built with FastAPI and modern web technologies.

## ✨ Features

- **🎯 Dual AI Models**: Fine-tuned local models + Groq cloud backup
- **🚀 Fast Generation**: Optimized for quick quest creation
- **🎨 Beautiful UI**: Modern glassmorphism design with animations
- **📱 Responsive**: Works on desktop and mobile devices
- **⚡ Real-time**: Instant quest generation with loading animations
- **🔄 Fallback System**: Automatic switching between AI models
- **📊 Status Monitoring**: Built-in health check endpoints



## 🖼️ UI Screenshots

<div align="center">
  <img src="https://github.com/parimal1009/Quest_Generator/blob/main/images/Screenshot%202025-07-24%20220227.png?raw=true" width="800" alt="Quest Generator UI 1" />
  <br><br>
  <img src="https://github.com/parimal1009/Quest_Generator/blob/main/images/Screenshot%202025-07-24%20220241.png?raw=true" width="800" alt="Quest Generator UI 2" />
  <br><br>
  <img src="https://github.com/parimal1009/Quest_Generator/blob/main/images/Screenshot%202025-07-24%20220306.png?raw=true" width="800" alt="Quest Generator UI 3" />
  <br><br>
  <img src="https://github.com/parimal1009/Quest_Generator/blob/main/images/Screenshot%202025-07-24%20220314.png?raw=true" width="800" alt="Quest Generator UI 4" />
</div>


## 🛠️ Tech Stack

- **Backend**: FastAPI, Python 3.8+
- **AI Models**: Hugging Face Transformers, Groq LLaMA
- **Frontend**: Vanilla JavaScript, Modern CSS
- **Training**: Custom fine-tuning pipeline
- **Deployment**: Uvicorn ASGI server

## 📦 Installation

### Prerequisites

- Python 3.8 or higher
- Git
- Virtual environment (recommended)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd text23d
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install fastapi uvicorn transformers torch datasets
   pip install langchain-groq  # Optional: for Groq integration
   ```

4. **Set up project structure**
   ```bash
   mkdir templates static models
   ```

5. **Add your HTML template**
   - Copy the provided HTML content to `templates/index.html`

6. **Set environment variables** (Optional)
   ```bash
   export GROQ_API_KEY="your_groq_api_key_here"
   ```

## 🚀 Usage

### Running the Application

```bash
# Default port (8000)
uvicorn app:app --reload

# Custom port (8087)
uvicorn app:app --reload --port 8087
```

Access the application at: `http://localhost:8087`

### Training Your Own Model

Run the fine-tuning script to create a custom quest generation model:

```bash
python train_model.py
```

**Training Configuration:**
- Model: FLAN-T5-Small (optimized for speed)
- Dataset: SQuAD (adapted for quest generation)
- Training time: ~15-30 minutes on modern hardware
- Output: `models/quest_generator/`

### API Endpoints

- **GET** `/` - Main web interface
- **POST** `/generate` - Generate quest from prompt
- **GET** `/status` - Health check and model status

### Example API Usage

```python
import requests

response = requests.post("http://localhost:8087/generate", 
                        json={"prompt": "Retrieve the Crystal of Eternal Light"})
quest = response.json()
print(quest["quest"])
```

## 🎮 Quest Generation Examples

**Input Prompts:**
- "Retrieve the Crystal of Eternal Light"
- "Rescue the captured dragon from the shadow realm"
- "Find the lost kingdom beneath the frozen mountains"
- "Stop the necromancer's army from rising"

**Sample Output:**
```
🌟 Quest: The Crystal of Eternal Light

Deep within the Whispering Caverns lies the legendary Crystal of Eternal Light, 
said to hold the power to banish darkness from the realm. The crystal has been 
stolen by the Shadow Cult and hidden in their fortress atop Mount Doomspire.

Your quest: Infiltrate the Shadow Fortress, overcome the cult's dark magic, 
and retrieve the crystal before the next eclipse, when its power will be lost forever.

Rewards: 1000 gold pieces, Crystal-enhanced weapon, Title: "Lightbringer"
```

## 🔧 Configuration

### Model Settings

```python
# app.py configuration
MODEL_PATH = "models/quest_generator"      # Local model path
GROQ_MODEL = "llama3-70b-8192"            # Groq model name
MAX_LENGTH = 200                          # Maximum quest length
```

### Training Parameters

```python
# train_model.py configuration
MODEL_NAME = "google/flan-t5-small"       # Base model
BATCH_SIZE = 16                           # Training batch size
EPOCHS = 1                                # Training epochs
TRAIN_SAMPLES = 5000                      # Training data size
```

## 📁 Project Structure

```
text23d/
├── app.py                 # FastAPI application
├── train_model.py         # Model training script
├── templates/
│   └── index.html        # Web interface
├── static/               # Static files (optional)
├── models/
│   └── quest_generator/  # Trained model files
├── README.md            # This file
└── requirements.txt     # Dependencies
```

## 🔍 Troubleshooting

### Common Issues

1. **Import Error: AutoModelForCSeq2SeqLM**
   - Fix: Change to `AutoModelForSeq2SeqLM` (remove the "C")

2. **Template Not Found**
   - Ensure `templates/index.html` exists
   - Check file permissions

3. **Model Loading Failed**
   - Run training script first: `python train_model.py`
   - Check `models/quest_generator/` directory exists

4. **Port Already in Use**
   - Use different port: `uvicorn app:app --reload --port 8088`
   - Kill existing process: `lsof -ti:8087 | xargs kill -9`

### Performance Tips

- **GPU Training**: Ensure CUDA is available for faster training
- **Memory Usage**: Reduce `BATCH_SIZE` if running out of RAM
- **Generation Speed**: Use smaller models for faster inference

## 🌟 Features Roadmap

- [ ] **Quest Difficulty Levels**: Easy, Medium, Hard, Epic
- [ ] **Character Classes**: Warrior, Mage, Rogue specific quests
- [ ] **World Building**: Consistent lore and locations
- [ ] **Quest Chains**: Multi-part adventures
- [ ] **Export Options**: PDF, JSON, plain text
- [ ] **User Accounts**: Save favorite quests
- [ ] **API Rate Limiting**: Production-ready deployment
- [ ] **Docker Support**: Containerized deployment

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Hugging Face** for the Transformers library
- **Groq** for fast inference API
- **FastAPI** for the excellent web framework
- **SQuAD Dataset** for training data
- **Fantasy RPG Community** for inspiration

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-username/fantasy-quest-generator/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/fantasy-quest-generator/discussions)
- **Email**: your-email@example.com

---

**Made with ❤️ for the fantasy RPG community**

*Generate epic quests, create legendary adventures, and bring your fantasy worlds to life!*
