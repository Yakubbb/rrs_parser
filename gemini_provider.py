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
                    "new_title": {"type": "string"},
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
        return f"""Ты выступаешь в роли группировщика новостей в аггрегаторе по категориям и событиям. 
                Ты получаешь набор новостей в формате {{title:заголовок новости, pubdate: дата публикации (иногда нужно учитывать при определения события)}}
                и ты должен сгруппировать по категориям и привязать их к конкретному событию. Из уже существующих событий есть:
                {', '.join(avalible_categories)} из уже существующих событий есть: {', '.join(avalible_events)},
                из уже существующих людей есть: {', '.join(avalible_persons)}.
                В случае если новость не подходит под событие или под категорию, ты можешь создать новую. Категорий у новости может быть несколько,
                но событие одно. Событие нужно для того, чтобы отслеживать разные источники по ОДНОМУ инфоповоду, поэтому событие не должно быть обобщенным,
                оно должно быть конкретным. Если новость состоит из высказывания какого-то спикера, то событие должно включать конкретную тему высказывания.
                В то же время категории могут иметь в себе новости одного типа, но по разным событиям. Если даже новость на английском, событие и категория
                должны быть на русском. От тебя я жду массив формата 
                {{title:заголовок новости который ты получил, new_title?:название если оригинальное
                название это ссылка event?:событие по этой новости, persons?:[массив упомянутых в новости людей],
                category:[массив категорий новости]}}.
                Нужно всё это для того, 
                чтобы по одному событию можно было получать новости с разных источников. Опять же, событие должно быть конкрентым. Если в заголовке 
                содержатся дата или место проведения события - это должно быть в событии, так как этот фильтр нужен для того чтобы новости 
                относились к КОНКРЕТНОМУ инфоповоду. Если новость не несёт в себе никакого события, то можешь не добавлять поле event, но категория 
                обязательна. Если название представляет из себя ссылку, то стоит попытаться извлечь из неё что-то осмысленное и вернуть это в new_title.
                Массив persons отвечает за упомянутых в новости людей. Если в новости был упомянут человек, то его имя и фамилия должны быть указаны в persons.
                Если в новости речь идёт о каком-то министре или ещё о ком, но не упоминается имя и фамилия, то вносить это в persons не нужно.
                persons содержит только имя и фамилию. иногда только фамилию, но без этих данных вносить в persons не имеет смысла. Если человек
                уже был упомянут ранее и имеется в списке существующих людей, то ты просто вносишь имя из списка в массив persons без изменений,
                дополнять его не нужно в этом случае."""
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
                        post.setTitle(cat.get('new_title',post.title))
                        post.setPersons(cat.get('persons',[]))
                        new_posts.append(post)
                        break
            
            return new_posts
                    
        else:
            print(Fore.RED + '[GEMINI] Ошибка: Нет ответа от модели или неверный формат ответа.' + Style.RESET_ALL)
            return []
                
                
            
            
            
            
        
        
        
        