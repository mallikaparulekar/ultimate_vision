from yt_dlp import YoutubeDL

def download_video_only(video_url, output_path='~/Desktop/CS/CS231N/ultiworld_vids/'):
    """
    Downloads the video-only stream (no audio) from the given YouTube URL.

    Parameters:
    - video_url (str): The URL of the YouTube video.
    - output_path (str): The output filename template (default is ~/Downloads/...).
    """
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]',  # Choose best video-only stream with mp4 extension
        'outtmpl': output_path,
        'postprocessors': [],  # No merging with audio
        'quiet': False,        # Set to True to suppress output
        'noplaylist': True, 
    }

    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])

# Example usage:
if __name__ == '__main__':
    url = "https://www.youtube.com/watch?v=xXjTszSINHU&list=PLvgVvH9p4IEGjQb50rn2R7E9cJ2IdfJ-r&index=20"
    download_video_only(url, output_path='~/Desktop/CS/CS231N/ultimate_fris_data_scraper/vids/%(title)s.%(ext)s')