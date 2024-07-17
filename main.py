import backend.src.resolute_cleaner as cleaner
import backend.src.resolute_scraper as scraper
import backend.src.resolute_extractor as extractor
import backend.src.normalizer as normalizer
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