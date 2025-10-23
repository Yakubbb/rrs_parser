import json
import os
from colorama import Fore, Style
from dotenv import load_dotenv
from mongo_connector import ParsedPost
from google.generativeai import GenerativeModel, configure

load_dotenv()
GEMINI_API_KEY = os.getenv('GOOGLE_API_KEY')
configure(api_key=GEMINI_API_KEY)


class GeminiProvider:

    GENERATION_CONFIG = {
        "response_mime_type": "application/json",
        "response_schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "event": {"type": "string"},
                    "category": {
                        "type": "array",
                        "items": {"type": "string"}
                        },
                    "persons": {
                        "type": "array",
                        "items": {"type": "string"}
                        }
                    },
                "required": ["title", "category"]
                }
            },
        }

    @staticmethod
    def create_system_prompt(avalible_events:list[str], avalible_categories:list[str], avalible_persons:list[str]) ->str :
        return f"""
Ты — ИИ-ассистент в новостном агрегаторе. Твоя задача — анализировать и структурировать входящий поток новостей.

Твоя основная цель: сгруппировать новости по **категориям** и **событиям**.

Тебе на вход будет подан список новостей в формате JSON: `{{title:заголовок, pubdate:дата}}`.
Ты должен вернуть JSON-массив, где для каждой новости указаны соответствующие поля.

---

### **Правила и определения:**

**1. Событие (event):**
- **Что это?** Событие — это КОНКРЕТНЫЙ инфоповод. Его цель — объединить новости из РАЗНЫХ источников, рассказывающие об ОДНОМ и том же происшествии, выступлении или явлении.
- **Каким оно должно быть?** МАКСИМАЛЬНО КОНКРЕТНЫМ. Не "Политика", а "Выступление [имя спикера] на саммите G20 15 ноября". Не "Авария", а "Крупное ДТП на трассе М-5 с участием бензовоза".
- **Что включать в название события?** Если в заголовке есть дата, место, участники, тема высказывания — всё это должно быть в названии события.
- **Когда не нужно?** Если новость не описывает конкретное событие (например, это еженедельный аналитический отчет, обзор рынка), то поле `event` можно не добавлять.
- **Существующие события:** Сначала проверь список уже существующих событий: `{', '.join(avalible_events)}`. Если новость подходит под одно из них — используй его. Если нет — создай новое по правилам выше.

**2. Категория (category):**
- **Что это?** Категория — это ОБЩАЯ тема новости. В отличие от события, категория может объединять новости о разных инфоповодах. У одной новости может быть несколько категорий.
- **Примеры:** "Политика", "Экономика", "Спорт", "Технологии", "Происшествия".
- **Существующие категории:** Проверь список: `{', '.join(avalible_categories)}`. Используй существующие, если подходят. Если нет — можешь создать новую, но старайся не плодить слишком много похожих.

**3. Упомянутые личности (persons):**
- **Кого добавлять?** Только людей, у которых указаны ИМЯ и/или ФАМИЛИЯ.
- **Кого НЕ добавлять?** Должности и титулы без имени ("министр", "глава компании") в это поле не вносятся.
- **Формат:** Массив строк, например: `["Иван Иванов", "Сергей Петров"]`.
- **Существующие личности:** Если упомянутый человек есть в списке `{', '.join(avalible_persons)}`, используй имя из списка.

**4. Заголовки (title):**
- **title:** Всегда содержит оригинальный заголовок новости, который ты получил.
**5. Язык:**
- Всегда отвечай на русском. Поля `event`, `category`, `persons` должны быть на РУССКОМ языке, даже если оригинальный заголовок новости на английском.

---

### **Итоговый формат ответа:**

Ты должен вернуть JSON-массив объектов. Каждый объект должен соответствовать одной новости и иметь следующую структуру:
`{{
  "title": "Оригинальный заголовок из входных данных",
  "event": "(опционально) Конкретное событие, к которому относится новость",
  "persons": "(опционально) [Массив имен и фамилий упомянутых людей]",
  "category": "[Массив из одной или нескольких категорий]"
}}`
Поле `category` является обязательным всегда.
"""

    @staticmethod
    def create_user_prompt(posts:list[ParsedPost]) ->str :
        return f'Твой набор новостей: {json.dumps([{"title":post.title, "pubdate":post.pubdate} for post in posts], ensure_ascii=False)}'

    @staticmethod
    def group_posts_with_gemini(user_prompt:str,system_prompt:str, posts_to_group:list[ParsedPost], model_name:str = 'gemini-2.5-flash-lite')->list[ParsedPost]:

        model = GenerativeModel(model_name, system_instruction=system_prompt,generation_config=GeminiProvider.GENERATION_CONFIG)
        response = model.generate_content(
            contents = user_prompt
        )
        if response.candidates and response.candidates[0].content.parts:
            gemini_output_text = response.candidates[0].content.parts[0].text
            print(f'[GEMINI] Ответ от модели: {gemini_output_text}')

            isParsed = False
            textBuff = gemini_output_text
            categories_titles = []

            while not isParsed and textBuff:
                try:
                    categories_titles = json.loads(textBuff)
                    isParsed = True
                    print("[GEMINI] Строка успешно разобрана.")
                except json.JSONDecodeError as e:
                    print(Fore.RED + f"[GEMINI] Ошибка при разборе JSON {e}" + Style.RESET_ALL)
                    potential_fix = textBuff + ']'
                    try:
                        categories_titles = json.loads(potential_fix)
                        isParsed = True
                        textBuff = potential_fix
                        print("[GEMINI] Строка успешно исправлена и разобрана.")
                    except json.JSONDecodeError:
                        textBuff = textBuff[:-1]

            new_posts = []

            for cat in categories_titles:
                for post in posts_to_group:
                    if post.title == cat.get('title',''):
                        post.setCategories(cat.get('category',[]))
                        post.setEvent(cat.get('event',''))
                        post.setPersons(cat.get('persons',[]))
                        new_posts.append(post)
                        break

            return new_posts

        else:
            print(Fore.RED + '[GEMINI] Ошибка: Нет ответа от модели или неверный формат ответа.' + Style.RESET_ALL)
            return []