from flask import Flask, request, jsonify, render_template, send_from_directory
from google import genai
from google.genai import types
import os
from dotenv import load_dotenv
import requests
import base64
import uuid
import logging

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")

# New Google GenAI SDK (google-genai), replacing the deprecated google-generativeai SDK.
client = genai.Client(api_key=GEMINI_API_KEY)
TEXT_MODEL = os.getenv("GEMINI_TEXT_MODEL", "gemini-3.6-flash")

os.makedirs("static", exist_ok=True)


def generate_image(prompt):
    """Generate an image with Stability AI if a Stability key is configured.
    Image generation is optional: recipe generation still succeeds if this fails.
    """
    if not STABILITY_API_KEY:
        logger.info("STABILITY_API_KEY not configured; skipping image generation.")
        return None

    url = "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image"
    headers = {
        "Authorization": f"Bearer {STABILITY_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    data = {
        "text_prompts": [{"text": prompt, "weight": 1}],
        "width": 1024,
        "height": 1024,
        "samples": 1,
        "style_preset": "photographic",
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        response.raise_for_status()
        return response.json()["artifacts"][0]["base64"]
    except Exception as e:
        logger.warning("Image generation failed; continuing without image: %s", e)
        return None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/saved-recipes')
def saved_recipes():
    return render_template('saved-recipes.html')


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/style.css')
def serve_css():
    return send_from_directory('.', 'style.css')


@app.route('/generate_recipe', methods=['POST'])
def generate_recipe():
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"success": False, "error": "No data provided"}), 400

        ingredients = data.get('ingredients', '').strip()
        language = data.get('language', 'English')
        prompt = data.get('prompt', '')
        strict_mode = data.get('strictMode', True)
        cuisine = data.get('cuisine', '')
        dietary_preferences = data.get('dietary_preferences', '')

        if not ingredients:
            return jsonify({"success": False, "error": "Please provide at least one ingredient"}), 400

        if strict_mode:
            mode_instruction = f"""
Create a recipe using ONLY these ingredients: {ingredients}.
Do NOT add extra food ingredients. If something is normally required but is not listed,
suggest a suitable alternative using one of the provided ingredients.
"""
        else:
            mode_instruction = f"""
Create a recipe using these primary ingredients: {ingredients}.
You may add reasonable common ingredients, while keeping the provided ingredients as the main focus.
"""

        full_prompt = f"""
You are a professional recipe creator.
Respond entirely in {language}.

{mode_instruction}

Additional requirements:
- Cuisine type: {cuisine or 'Any'}
- Dietary preferences: {dietary_preferences or 'None specified'}
- User notes: {prompt or 'None'}

Include:
1. A creative recipe name
2. Ingredients with measurements
3. Clear step-by-step instructions
4. Cooking time
5. Serving size
6. Optional tips or variations

Make the recipe practical and easy to follow.
"""

        logger.info("Generating recipe using %s", TEXT_MODEL)
        response = client.models.generate_content(
            model=TEXT_MODEL,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.95,
                max_output_tokens=4096,
            ),
        )

        recipe_text = response.text
        image_url = None

        image_prompt = (
            f"Professional food photograph of a {cuisine or 'home-style'} dish made with "
            f"{ingredients}; {dietary_preferences or 'general'} dietary preference; "
            "appetizing, realistic, well-lit, plated beautifully, no text"
        )

        image_data = generate_image(image_prompt)
        if image_data:
            image_filename = f"recipe_{uuid.uuid4().hex}.png"
            image_path = os.path.join("static", image_filename)
            with open(image_path, "wb") as f:
                f.write(base64.b64decode(image_data))
            image_url = f"/static/{image_filename}"

        return jsonify({
            "success": True,
            "recipe": recipe_text,
            "image_url": image_url,
        })

    except Exception as e:
        logger.exception("Recipe generation error")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
