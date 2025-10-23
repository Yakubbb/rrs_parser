import asyncio
import random
import time
from rss import parse_opml_and_rss
from telegram import parse_tg
from mongo_connector import ParsedPost, get_avalible_categories, get_avalible_events, get_avalible_persons, select_only_new_posts,insert_new_posts
from gemini_provider import GeminiProvider
from concurrent.futures import ThreadPoolExecutor
from colorama import Fore, Style
import schedule


def parse_all_posts() -> list[ParsedPost]:
    start_time = time.time()
    
    opml_file = 'subscriptions.opml'
    json_file = 'tg.json'
    
        
    parsed_tg  = asyncio.run(parse_tg(json_file))
    parsed_rss = asyncio.run(parse_opml_and_rss(opml_file))
    
    
    parsed_posts = list(set(parsed_tg + parsed_rss))
    
    sources = set([post.source for post in parsed_posts])
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    minutes = int(elapsed_time // 60)
    seconds = int(elapsed_time % 60)
    
    print('\n\n')
    print(f'[MAIN] Время на парсинг: {minutes} мин. {seconds} сек.')
    print(f'[MAIN] Всего новостей получено с источников: {len(parsed_posts)}\n Из rss: {len(parsed_rss)}\n Из тг: {len(parsed_tg)}\n Количество источников: {len(sources)}')
    
    return parsed_posts

def update_global_queue():
    global POST_QUEUE
    parsed_posts = parse_all_posts()
    POST_QUEUE = select_only_new_posts(parsed_posts)
    random.shuffle(POST_QUEUE)

POST_QUEUE = []

def main_loop():
    global POST_QUEUE
    try:
        if len(POST_QUEUE) < 10:
            update_global_queue()
        print(Fore.MAGENTA +f'[MAIN] Новостей в очереди: {len(POST_QUEUE)}' + Style.RESET_ALL)
        selected_posts = POST_QUEUE[-100:]
        
        avalible_categories = get_avalible_categories()
        avalible_events = get_avalible_events()
        avalible_persons = get_avalible_persons()
        
        system_prompt = GeminiProvider.create_system_prompt(avalible_events,avalible_categories,avalible_persons)
        user_prompt = GeminiProvider.create_user_prompt(selected_posts)
        gemini_answer = GeminiProvider.group_posts_with_gemini(user_prompt,system_prompt,selected_posts)
        
        if len(gemini_answer) > 0:
            insert_new_posts(gemini_answer)
            for post in selected_posts: POST_QUEUE.remove(post)
    except Exception as e:
        print(Fore.RED + f'[MAIN] Ошибка в основном цикле: {e}' + Style.RESET_ALL)
        
schedule.every(1).minute.do(main_loop)

if __name__ == "__main__":
    while True:
        schedule.run_pending()
        time.sleep(60) 
    


    
    
    
    
