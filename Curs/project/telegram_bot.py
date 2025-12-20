import logging
import os
import time
import joblib
import numpy as np
import cv2
import tensorflow as tf
from skimage import feature
from PIL import Image
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackQueryHandler, ContextTypes
import config

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

# --- Model Loading ---
try:
    logger.info("Loading models and metadata...")
    # Model 1: CNN
    CNN_MODEL = tf.keras.models.load_model('models/best_cpu_model.h5')
    
    # Model 2: Traditional (HOG + SVM)
    TRADITIONAL_MODEL_DATA = joblib.load('models/traditional_model_hog.pkl')
    TRADITIONAL_SVM = TRADITIONAL_MODEL_DATA['svm']
    TRADITIONAL_SCALER = TRADITIONAL_MODEL_DATA['scaler']
    TRADITIONAL_HOG_PARAMS = TRADITIONAL_MODEL_DATA['hog_params']

    # Model 3: Random Forest (HOG + Color)
    RF_MODEL_DATA = joblib.load('models/random_forest_model.pkl')
    RF_MODEL = RF_MODEL_DATA['rf']
    RF_SCALER = RF_MODEL_DATA['scaler']
    RF_HOG_PARAMS = RF_MODEL_DATA['hog_params']
    RF_COLOR_BINS = RF_MODEL_DATA.get('color_bins', 16)

    # Metadata (assuming all models use the same labels)
    META_DATA = joblib.load('models/euro_sat_cpu_model_meta.pkl')
    LABEL_MAP = META_DATA['label_map']
    # Create a reverse map from index to label name
    REVERSE_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}
    
    TARGET_SIZE = (64, 64)
    logger.info("Models and metadata loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    CNN_MODEL = None
    # Add other model variables here to avoid NameError if loading fails

# --- Prediction Functions ---

def predict_cnn(image_path):
    """Predicts class for an image using the CNN model."""
    start_time = time.time()
    
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=TARGET_SIZE)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    
    predictions = CNN_MODEL.predict(img_array)
    predicted_class_index = np.argmax(predictions[0])
    class_name = REVERSE_LABEL_MAP[predicted_class_index]
    
    inference_time = time.time() - start_time
    return class_name, inference_time

def predict_traditional(image_path):
    """Predicts class for an image using the Traditional (HOG+SVM) model."""
    start_time = time.time()
    
    img = cv2.imread(image_path)
    img = cv2.resize(img, TARGET_SIZE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    hog_feat = feature.hog(gray, **TRADITIONAL_HOG_PARAMS)
    hog_feat_scaled = TRADITIONAL_SCALER.transform(hog_feat.reshape(1, -1))
    
    predicted_class_index = TRADITIONAL_SVM.predict(hog_feat_scaled)[0]
    class_name = REVERSE_LABEL_MAP[predicted_class_index]
    
    inference_time = time.time() - start_time
    return class_name, inference_time

def predict_random_forest(image_path):
    """Predicts class for an image using the Random Forest model."""
    start_time = time.time()

    # Load and normalize image
    img = Image.open(image_path).resize(TARGET_SIZE, Image.Resampling.LANCZOS)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img).astype('float32') / 255.0

    # HOG features
    gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
    hog_features = feature.hog(gray, **RF_HOG_PARAMS)

    # Color features
    hist_r = np.histogram(img_array[:,:,0], bins=RF_COLOR_BINS, range=(0, 1))[0]
    hist_g = np.histogram(img_array[:,:,1], bins=RF_COLOR_BINS, range=(0, 1))[0]
    hist_b = np.histogram(img_array[:,:,2], bins=RF_COLOR_BINS, range=(0, 1))[0]
    stats = [
        np.mean(img_array[:,:,0]), np.std(img_array[:,:,0]),
        np.mean(img_array[:,:,1]), np.std(img_array[:,:,1]),
        np.mean(img_array[:,:,2]), np.std(img_array[:,:,2]),
        np.mean(img_array), np.std(img_array),
        np.max(img_array) - np.min(img_array)
    ]
    color_features = np.concatenate([hist_r, hist_g, hist_b, stats])

    # Combine and scale
    combined_features = np.hstack([hog_features, color_features])
    scaled_features = RF_SCALER.transform(combined_features.reshape(1, -1))
    
    predicted_class_index = RF_MODEL.predict(scaled_features)[0]
    class_name = REVERSE_LABEL_MAP[predicted_class_index]

    inference_time = time.time() - start_time
    return class_name, inference_time


# --- Telegram Handlers ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Привет, {user.mention_html()}! Отправь мне фотографию для классификации.",
    )

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles photos sent by the user."""
    photo_file = await update.message.photo[-1].get_file()
    
    # We need a unique path for each user's photo
    file_path = os.path.join('temp_images', f'{update.effective_user.id}_{photo_file.file_unique_id}.jpg')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    await photo_file.download_to_drive(file_path)
    
    # Store the file path for the callback query
    context.user_data['photo_path'] = file_path
    
    keyboard = [
        [InlineKeyboardButton("Модель 1 (CNN)", callback_data='1')],
        [InlineKeyboardButton("Модель 2 (Traditional)", callback_data='2')],
        [InlineKeyboardButton("Модель 3 (Random Forest)", callback_data='3')],
        [InlineKeyboardButton("Все модели", callback_data='all')],
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text("Отлично! Теперь выберите модель для обработки:", reply_markup=reply_markup)

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Parses the CallbackQuery and runs the chosen model."""
    query = update.callback_query
    await query.answer() # Acknowledge the button press
    
    choice = query.data
    image_path = context.user_data.get('photo_path')

    if not image_path or not os.path.exists(image_path):
        await query.edit_message_text(text="Ошибка: не могу найти ваше изображение. Пожалуйста, отправьте его снова.")
        return

    await query.edit_message_text(text=f"Обрабатываю изображение с помощью '{choice}'...")

    response_text = ""
    try:
        if choice == '1':
            name, infer_time = predict_cnn(image_path)
            response_text = f"🖼️ Модель CNN определила: **{name}**\n⏱️ Время обработки: {infer_time:.4f} сек."
        elif choice == '2':
            name, infer_time = predict_traditional(image_path)
            response_text = f"🖼️ Модель Traditional (HOG+SVM) определила: **{name}**\n⏱️ Время обработки: {infer_time:.4f} сек."
        elif choice == '3':
            name, infer_time = predict_random_forest(image_path)
            response_text = f"🖼️ Модель Random Forest определила: **{name}**\n⏱️ Время обработки: {infer_time:.4f} сек."
        elif choice == 'all':
            name1, time1 = predict_cnn(image_path)
            name2, time2 = predict_traditional(image_path)
            name3, time3 = predict_random_forest(image_path)
            response_text = (
                "**Сравнение моделей:**\n\n"
                f"1. **CNN**:\n   - Результат: **{name1}**\n   - Время: {time1:.4f} сек.\n\n"
                f"2. **Traditional (HOG+SVM)**:\n   - Результат: **{name2}**\n   - Время: {time2:.4f} сек.\n\n"
                f"3. **Random Forest**:\n   - Результат: **{name3}**\n   - Время: {time3:.4f} сек."
            )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        response_text = f"Произошла ошибка при обработке изображения: {e}"

    await query.edit_message_text(text=response_text, parse_mode='Markdown')
    
    # Clean up the downloaded file
    try:
        os.remove(image_path)
    except Exception as e:
        logger.warning(f"Could not remove temp file {image_path}: {e}")


def main() -> None:
    """Start the bot."""
    if not all([CNN_MODEL]):
        logger.critical("Could not load one or more models. Bot cannot start.")
        return
        
    application = Application.builder().token(config.BOT_TOKEN).build()

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    application.add_handler(CallbackQueryHandler(button_handler))

    # Create a directory for temporary images
    os.makedirs('temp_images', exist_ok=True)
    
    # Start the Bot
    logger.info("Starting bot...")
    application.run_polling()


if __name__ == "__main__":
    main()