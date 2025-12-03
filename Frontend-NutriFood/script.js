const fileInput = document.getElementById('fileElem');
const previewImg = document.getElementById('preview-img');
const placeholder = document.getElementById('placeholder');
const btnPredict = document.getElementById('btn-predict');
const chatHistory = document.getElementById('chat-history');
const userInput = document.getElementById('user-input');

let currentContext = null; 

function handleFiles(files) {
    if (files.length > 0) {
        const file = files[0];
        const reader = new FileReader();
        reader.onload = (e) => {
            previewImg.src = e.target.result;
            previewImg.classList.remove('hidden');
            placeholder.classList.add('hidden');
            btnPredict.disabled = false;
        }
        reader.readAsDataURL(file);
    }
}

async function uploadImage() {
    const file = fileInput.files[0];
    if (!file) return;

    btnPredict.disabled = true;
    btnPredict.innerText = "Analizando...";
    addBotMessage("⏳ Analizando tu imagen, dame un segundo...");

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            body: formData
        });

        if (response.ok) {
            const data = await response.json();
            
            currentContext = {
                food: data.predictions[0].label.replace(/_/g, ' '),
                confidence: (data.predictions[0].confidence * 100).toFixed(1),
                nutrients: data.nutrition?.result?.nutriments || {}
            };

            document.getElementById('quick-result').classList.remove('hidden');
            document.getElementById('food-label').innerText = currentContext.food.toUpperCase();
            document.getElementById('conf-fill').style.width = currentContext.confidence + "%";
            document.getElementById('conf-text').innerText = currentContext.confidence + "% Certeza";

            const cal = Math.round(currentContext.nutrients['energy-kcal_100g'] || 0);
            addBotMessage(`✅ ¡Listo! He detectado **${currentContext.food}**. \n\nTiene aproximadamente **${cal} kcal** por cada 100g. \n\n¿Qué más te gustaría saber?`);

        } else {
            addBotMessage("❌ Error en el servidor. Verifica Docker.");
        }
    } catch (error) {
        console.error(error);
        addBotMessage("❌ Error de conexión.");
    } finally {
        btnPredict.disabled = false;
        btnPredict.innerText = "Analizar Alimento";
    }
}

function handleEnter(e) {
    if (e.key === 'Enter') sendMessage();
}

function sendMessage() {
    const text = userInput.value.trim();
    if (!text) return;

    addUserMessage(text);
    userInput.value = "";

    setTimeout(() => {
        processResponse(text.toLowerCase());
    }, 600);
}

function processResponse(text) {
    if (!currentContext) {
        addBotMessage("Primero sube una imagen para saber de qué hablamos. 📸");
        return;
    }

    const n = currentContext.nutrients;
    const food = currentContext.food;

    if (text.includes("caloria") || text.includes("caloría")) {
        const val = n['energy-kcal_100g'] || "?";
        addBotMessage(`🔥 **${food}**: ${val} kcal / 100g.`);
    
    } else if (text.includes("proteina") || text.includes("proteína")) {
        const val = n['proteins_100g'] || 0;
        addBotMessage(`💪 Proteínas: **${val}g**.`);
    
    } else if (text.includes("grasa")) {
        const val = n['fat_100g'] || 0;
        addBotMessage(`g Grasa total: **${val}g**.`);
    
    } else if (text.includes("carbo")) {
        const val = n['carbohydrates_100g'] || 0;
        addBotMessage(`🍞 Carbohidratos: **${val}g**.`);

    } else if (text.includes("gracias")) {
        addBotMessage("¡De nada! 🥗");

    } else {
        addBotMessage(`Intenta preguntar por "calorías", "proteínas" o "grasas".`);
    }
}

// --- FUNCIÓN NUEVA: LIMPIAR CHAT ---
function clearChat() {
    // 1. Borrar todo el contenido
    chatHistory.innerHTML = '';
    
    // 2. Agregar mensaje de bienvenida de nuevo
    addBotMessage("✨ Chat limpio. ¿En qué puedo ayudarte ahora?");
    
    // Opcional: Si quieres olvidar la comida actual, descomenta esto:
    // currentContext = null;
    // document.getElementById('quick-result').classList.add('hidden');
}

function getTime() {
    const now = new Date();
    return now.getHours() + ":" + String(now.getMinutes()).padStart(2, '0');
}

function addBotMessage(htmlText) {
    const div = document.createElement('div');
    div.className = 'message bot-msg';
    div.innerHTML = `<div class="bubble">${htmlText.replace(/\n/g, '<br>')}</div><span class="timestamp">${getTime()}</span>`;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}

function addUserMessage(text) {
    const div = document.createElement('div');
    div.className = 'message user-msg';
    div.innerHTML = `<div class="bubble">${text}</div><span class="timestamp">${getTime()}</span>`;
    chatHistory.appendChild(div);
    chatHistory.scrollTop = chatHistory.scrollHeight;
}