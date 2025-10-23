import asyncio
import aiohttp
import feedparser
import listparser
from random import shuffle
from mongo_connector import ParsedPost
from colorama import Fore, Style

async def fetch_and_parse_feed(session: aiohttp.ClientSession, feed_info: object) -> list[ParsedPost]:
    feed_url = feed_info.url
    feed_title = feed_info.title if feed_info.title else feed_url
    local_parsed_list = []
    
    print(Style.RESET_ALL + f"[RSS] Лента: {feed_title} ({feed_url})")

    try:
        async with session.get(feed_url, timeout=15) as response:
            response.raise_for_status()
            content = await response.read()
            
        feed = feedparser.parse(content)

        if hasattr(feed, 'bozo') and feed.bozo:
            print(Fore.RED + f"[RSS] Внимание: Обнаружены проблемы при парсинге: {feed_url} | {feed.bozo_exception}" + Style.RESET_ALL)

        if not feed.entries:
            print(Fore.YELLOW + f"[RSS] Нет записей в этой ленте: {feed_url}" + Style.RESET_ALL)
            return []
            
        for entry in feed.entries:
            title = entry.title if hasattr(entry, 'title') else "Без заголовка"
            link = entry.link if hasattr(entry, 'link') else "Нет ссылки"
            published = entry.published if hasattr(entry, 'published') else "Неизвестно"
            
           
            image_url = None
            if hasattr(entry, 'enclosures') and entry.enclosures:
                for enclosure in entry.enclosures:
                    if 'image' in enclosure.get('type', ''):
                        image_url = enclosure.get('url')
                        #print(image_url)
                        if image_url:
                            break
                        
            local_parsed_list.append(ParsedPost(feed_title, title, published, link, feed_url, image_url))
                    
    except asyncio.TimeoutError:
        print(Fore.RED + f"[RSS] Таймаут при подключении к ленте: {feed_url}" + Style.RESET_ALL)
    except aiohttp.ClientError as e:
        print(Fore.RED + f"[RSS] Сетевая ошибка при доступе к ленте '{feed_url}': {e}" + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"[RSS] Ошибка при обработке ленты '{feed_url}': {e}"+ Style.RESET_ALL)
        
    return local_parsed_list


async def parse_opml_and_rss(opml_file_path: str) -> list[ParsedPost]:
    try:
        with open(opml_file_path, 'r', encoding='utf-8') as f:
            opml_data = listparser.parse(f.read())
    except FileNotFoundError:
        print(Fore.RED + f"[RSS] Ошибка: Файл OPML '{opml_file_path}' не найден." + Style.RESET_ALL)
        return []
    except Exception as e:
        print(Fore.RED + f"[RSS] Ошибка при парсинге OPML-файла: {e}" + Style.RESET_ALL)
        return []

    if not opml_data.feeds:
        print(Fore.YELLOW + "[RSS] В OPML-файле не найдено RSS-лент." + Style.RESET_ALL)
        return []
        
    print(Style.RESET_ALL + f"[RSS] Найдено {len(opml_data.feeds)} RSS-лент в OPML-файле." + Style.RESET_ALL)
    
    all_parsed_posts = []
    feeds = opml_data.feeds
    shuffle(feeds)
    

    async with aiohttp.ClientSession() as session:
        tasks = [fetch_and_parse_feed(session, feed_info) for feed_info in feeds]
        results = await asyncio.gather(*tasks)

        for post_list in results:
            all_parsed_posts.extend(post_list)

    return all_parsed_posts

