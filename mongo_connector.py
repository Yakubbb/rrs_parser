import os
from datetime import datetime, timedelta
from colorama import Fore, Style
from dotenv import load_dotenv
from pymongo import MongoClient
from bson.objectid import ObjectId


load_dotenv()
uri = os.getenv("MONGODB_URI")

client = MongoClient(uri)
db = client["rss-parser"]
collection = db["news"]

load_dotenv()
uri = os.getenv("MONGODB_URI")


class ParsedPost:
    def __init__(self, source:str, title:str, pubdate, link_html:str, link_xml:str, image_url:str=None):
        self.source = source
        self.title = title
        self.pubdate = pubdate
        self.link_html = link_html
        self.link_xml = link_xml
        self.persons = []
        self.image_url = image_url
        
    def setCategories(self, categories:list[str]):
        self.categories = categories if len(categories)>0 else []
        
    def setPersons(self, persons:list[str]):
        self.persons = persons if len(persons)>0 else []

    def setEvent(self, event:str):
        self.event = event
            
    def setTitle(self, new_title:str):
        self.title = new_title



def select_only_new_posts(posts:list[ParsedPost]):
    try:
        
        collection_items = get_all_posts()
        existing_links = []
        
        for item in collection_items:
            if 'link_html' not in item:
                print(Fore.YELLOW  + f"[MONGO] Предупреждение: документ с _id {item.get('_id')} не содержит поле 'link_html'." + Style.RESET_ALL)
            else:
                existing_links.append(item['link_html'])
                
        not_existing_posts = [post for post in posts if post.link_html not in existing_links]
        print(f"[MONGO] Из {len(posts)} новостей  новых только: {len(not_existing_posts)}")
        
    except Exception as e:
        print(Fore.RED + f"[MONGO] Ошибка при парсинге из MongoDB: {e}" + Style.RESET_ALL)
        not_existing_posts = []
    
    return not_existing_posts

def insert_new_posts(posts_with_categories:list[ParsedPost]):
    
    expires_at = datetime.now() + timedelta(days=1)
    
    new_posts = select_only_new_posts(posts_with_categories)
    posts_to_insert =[]
    
    
    for post in new_posts:
        posts_to_insert.append(
            {
                "title":post.title,
                "from":post.source,
                "pubdate":post.pubdate,
                "link_html":post.link_html,
                "link_xml":post.link_xml,
                "category":post.categories,
                "event":post.event,
                "expires_at":expires_at,
                "persons":post.persons,
                "image_url":post.image_url
            }
        )
    
    result = collection.insert_many(posts_to_insert)
    
    inserted_posts = []
    for i, post in enumerate(posts_to_insert):
        post['_id'] = str(result.inserted_ids[i])
        inserted_posts.append(post)
        
    return inserted_posts

def get_all_posts():
    all_posts = list(collection.find())
    return all_posts

def get_avalible_categories():
    all_categories = []
    for doc in collection.find({}, {"category": 1}):
        if "category" in doc and doc["category"]:
            all_categories.extend(doc["category"])     
    return list(set(all_categories))

def get_avalible_events():
    events = []
    for doc in collection.find({"event": {"$exists": True}}, {"event": 1}):
        if "event" in doc and doc["event"]:
            events.append(doc["event"])
    return list(set(events))

def get_avalible_persons():
    all_categories = []
    for doc in collection.find({}, {"persons": 1}):
        if "persons" in doc and doc["persons"]:
            all_categories.extend(doc["persons"])     
    return list(set(all_categories))



