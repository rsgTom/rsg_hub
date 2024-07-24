import backend.src.modules.rsg_blog.resolute_cleaner as cleaner
import backend.src.modules.rsg_blog.resolute_scraper as scraper
import backend.src.modules.rsg_blog.resolute_extractor as extractor
import backend.src.modules.rsg_blog.normalizer as normalizer
import asyncio

async def main():
    """
    A coroutine that calls the main functions of resolute_cleaner and resolute_scraper.
    """
    await scraper.main()
    await extractor.main()
    await normalizer.main()
    # await cleaner.main()
    
    
if __name__ == "__main__":
    asyncio.run (main())