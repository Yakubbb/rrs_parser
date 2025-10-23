import asyncio
import json
import aiohttp
from random import shuffle
import aiofiles
from bs4 import BeautifulSoup
from mongo_connector import ParsedPost
from colorama import Fore, Style

async def fetch_and_parse_url(session: aiohttp.ClientSession, url: str) -> list[ParsedPost]:
    local_parsed_list = []
    try:
        async with session.get(url, timeout=15) as response:
            response.raise_for_status()
            html = await response.text()

        soup = BeautifulSoup(html, 'lxml')
        message_blocks = soup.find_all('div', class_=['tgme_widget_message'], attrs={'data-post': True})
        
        print(Style.RESET_ALL + f'[TELEGRAM] Парсинг канала ({url})')
        
        for block in message_blocks:
            data_post_value = block.get('data-post')
            post_text_element = block.find('div', class_=['tgme_widget_message_text', 'js-message_text'])
            post_date_element = block.find('a', class_='tgme_widget_message_date')
            
            post_datetime = ""
            if post_date_element and post_date_element.find('time'):
                post_datetime = post_date_element.find('time').get('datetime')

            if post_text_element and data_post_value:
                post_text = post_text_element.get_text(strip=True)
                link = f"https://t.me/{data_post_value}"
                feed_url = url
                channel_name = data_post_value.split('/')[0]
                local_parsed_list.append(ParsedPost(channel_name, post_text, post_datetime, link, feed_url))
            else:
                post_link_for_log = f"https://t.me/{data_post_value}" if data_post_value else "Неизвестный пост"
                print(Fore.YELLOW + f'[TELEGRAM][{url}] Не удалось найти текст поста. ({post_link_for_log})')
    
    except asyncio.TimeoutError:
        print(Fore.RED + f"[TELEGRAM] Таймаут при подключении к {url}")
    except aiohttp.ClientError as e:
        print(Fore.RED + f"[TELEGRAM] Сетевая ошибка при доступе к {url}: {e}")
    except Exception as e:
        print(Fore.RED + f"[TELEGRAM] Произошла непредвиденная ошибка при обработке {url}: {e}")
        
    return local_parsed_list

async def parse_tg(json_file_path: str) -> list[ParsedPost]:
    try:
        async with aiofiles.open(json_file_path, 'r', encoding='utf-8') as f:
            content = await f.read()
            urls:list[str] = json.loads(content)
    except FileNotFoundError:
        print(Fore.RED + f"[TELEGRAM] Ошибка: Файл {json_file_path} не найден.")
        return []
    except json.JSONDecodeError:
        print(Fore.RED + f"[TELEGRAM] Ошибка: Неверный формат файла {json_file_path}.")
        return []

    all_parsed_posts = []
    shuffle(urls)
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_parse_url(session, url) for url in  urls]
        results = await asyncio.gather(*tasks)
        
        for post_list in results:
            all_parsed_posts.extend(post_list)
            
    return all_parsed_posts
