from collections import defaultdict
import csv
from datetime import date
import os
import json
import logging
import sys
import statistics
import polars as pl
import string_utils
from github import Github
from groq import Groq
from PIL import Image, ImageDraw, ImageFont

config = {
    'github': {
        'repo_name': os.getenv('GITHUB_REPOSITORY'),
        'token': os.getenv('GITHUB_TOKEN'),
    },
    'groq_api_key': os.getenv('GROQ_API_KEY'),
}


def load_config():
    """Builds global configuration."""
    try:
        with open('assets/config.json', 'r', encoding='utf-8') as f_config:
            config_json = json.load(f_config)
    except FileNotFoundError:
        logging.error("assets/config.json not found.")
        config_json = {}
    except json.JSONDecodeError:
        logging.error("Could not decode JSON from assets/config.json.")
        config_json = {}
    global config
    config = {**config, **config_json}


def fetch_recent_haikus(count=50):
    if os.path.exists(config['paths']['history']):
        return pl.read_csv(config['paths']['history']) \
                    .filter(pl.col('is_winner') == 'true') \
                    .select('haiku') \
                    .tail(count) \
                    .to_dict(as_series=False)['haiku']
    return {}


def groq(model: str, prompt: str, sysprompt='', temperature=0.8) -> str:
    """'Groq' the provided model with the specified prompt, system prompt, and temperature."""
    client = Groq(api_key=config['groq_api_key'])
    messages = []
    if sysprompt:
        messages.append({"role": "system", "content": sysprompt})
    messages.append({"role": "user", "content": prompt})
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_completion_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content


def get_rated_haikus(recent_winning_haikus):
    """Generates haikus and uses AI models to rate them on a specified scale."""
    contextual_prompt = ''
    if recent_winning_haikus:
        contextual_prompt = f'{config['prompts']['generate_haiku']} These are previously generated haikus: {recent_winning_haikus}. You must avoid repeating the themes and specific phrases from these examples.'
    
    prompt = contextual_prompt or config['prompts']['generate_haiku']
    
    haikus = [
        groq(
            config['models'][model]['technical_name'],
            prompt,
        ) for model in config['models']
    ]
    haikus_rated = [
        groq(
            config['models'][model]['technical_name'],
            json.dumps(haikus),
            sysprompt=config['prompts']['rate_haiku'],
        )
        for model in config['models']
    ]
    return haikus, haikus_rated


def aggregate_scores(haikus_rated_list):
    """Calculate mean and std scores for rated haikus."""
    mas, std = defaultdict(list), defaultdict(list)
    for haikus_rated in haikus_rated_list:
        for haiku_rated in haikus_rated:
            mas[haiku_rated['haiku_index']].append(haiku_rated['overall_score'])
            std[haiku_rated['haiku_index']].append(haiku_rated['overall_score'])
    for idx in mas:
        mas[idx] = sum(mas[idx]) / len(mas[idx])
        std[idx] = statistics.stdev(std[idx])
    return mas, std


def add_text_to_gif(
    text,
    gif_path,
    output_path,
    position=(10, 10),
    font_size=20,
    font_color_rgba=(0, 0, 0, 255),
):
    """..."""
    try:
        img = Image.open(gif_path)
    except FileNotFoundError:
        logging.error('Input GIF not found at %s', config["paths"]["gif"])
        return
    except Exception:
        logging.exception('Error opening GIF')
        return

    frames = []
    durations = []
    loop_info = img.info.get('loop', 0)

    try:
        try:
            font = ImageFont.truetype(font=config['paths']['font'], size=font_size)
        except IOError:
            print(
                f"Warning: Font '{config['paths']['font']}' not found. Using default."
            )
            font = ImageFont.load_default()

        for i in range(img.n_frames):
            img.seek(i)
            frame = img.convert('RGBA')
            draw = ImageDraw.Draw(frame)
            draw.text(
                position,
                string_utils.replace_br_with_newlines(text),
                font=font,
                fill=font_color_rgba,
            )
            frames.append(frame)
            durations.append(img.info.get('duration', 100))

        if not frames:
            logging.error('No frames processed.')
            return

        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=durations,
            loop=loop_info,
            disposal=2,
        )
        logging.info('GIF saved to %s', config['paths']['gif'])

    except Exception:
        logging.exception('Error during processing')


def generate_md_table(haikus, stats, means, stds, win_idx):
    """Generates a Markdown table from data[]."""

    def build_header(models) -> str:
        header = '| Haiku | Generated By'
        for model in models:
            header = f'{header} | Rated by `{model}`'
        header = f'{header} | Mean Score | Std Dev | Status |'
        return header

    if not stats:
        return ''

    models = [config['models'][model]['display_name'] for model in config['models']]
    header = build_header(models)
    separator = '| :---------------------------------------------- | :----------- | :----------------- | :---------------- | :----------------- | :--------- | :--------- | :-------- |'
    body = ''
    haikus = [string_utils.replace_newlines_with_br(haiku) for haiku in haikus]

    # as models x haikus is square (3x3)
    for i in range(len(stats)):
        body += f'*{haikus[i].strip()}* | {models[i]}'
        for stat in stats:
            body = f'{body} | {round(stat[i]['overall_score'], 2)} / 5'
        body += f'| {round(means[i], 2)}'
        body += f' | {round(stds[i], 4)}'
        if win_idx == i:
            body += ' | 🏆 Winner |'
        else:
            body += ' |  |'
        body += '\n'

    return f'\n\n{header}\n{separator}\n{body}'


def format_as_csv_rows(haikus, stats, means, stds, win_idx, file_exists):
    """Formats results into CSV rows."""

    def header() -> str:
        header = ['date', 'haiku', 'generator']
        for model in list(config['models']):
            header += [f'rating_by_{config['models'][model]['csv_name']}']
        header += ['mean_score', 'std', 'is_winner']
        return header

    models = [config['models'][model]['display_name'] for model in config['models']]
    haikus = [string_utils.replace_newlines_with_br(haiku) for haiku in haikus]

    csv_rows = []
    for i in range(len(stats)):
        body = [f'{str(date.today())}', f'{haikus[i].strip()}', f'{models[i]}']
        for stat in stats:
            body += [f'{round(stat[i]['overall_score'], 2)}']
        body += [f'{round(means[i], 2)}']
        body += [f'{round(stds[i], 4)}']
        if win_idx == i:
            body += ['TRUE']
        else:
            body += ['FALSE']
        csv_rows.append(body)

    if not file_exists:
        csv_rows.insert(0, header())

    return csv_rows


def update_readme(haikus, stats, means, stds, win_idx):
    """Updates the detailed results section of the README."""
    try:
        try:
            with open(config['paths']['readme'], 'r', encoding='utf-8') as f:
                readme_content = f.read()

                stats_marker_start = '<div id="stats_marker"></div>'
                stats_marker_end = '</details>'
                stats_start_pos = readme_content.find(stats_marker_start)
                stats_start_pos += len(stats_marker_start)
                stats_end_pos = readme_content.find(stats_marker_end, stats_start_pos)
                new_stats = generate_md_table(haikus, stats, means, stds, win_idx)

                readme_content_updated = (
                    readme_content[:stats_start_pos]
                    + new_stats
                    + readme_content[stats_end_pos:]
                )

                with open(config['paths']['readme'], 'w', encoding='utf-8') as f:
                    f.write(readme_content_updated)
                    logging.info(
                        "Successfully updated %s with the new section content.",
                        config['paths']['readme'],
                    )

                    return readme_content_updated

            return ''

        except FileNotFoundError:
            logging.exception('README file not found at %s', config['paths']['readme'])
            return

    except Exception:
        logging.exception("Error updating README")


def update_history(haikus, stats, means, stds, win_idx):
    """Record daily haikus in history CSV."""
    file_exists = os.path.exists(config['paths']['history'])
    with open(
        config['paths']['history'], 'a', newline='', encoding='utf-8'
    ) as history_file:
        writer = csv.writer(history_file)
        writer.writerows(
            format_as_csv_rows(
                haikus,
                stats,
                means,
                stds,
                win_idx,
                file_exists,
            )
        )


if __name__ == "__main__":
    load_config()

    if not config['github']['token']:
        logging.error('GITHUB_TOKEN not found. Cannot interact with GitHub API.')
        sys.exit(1)
    if not config['github']['repo_name']:
        logging.error('GITHUB_REPOSITORY not found')
        sys.exit(1)

    g = Github(config['github']['token'])
    repo = g.get_repo(config['github']['repo_name'])

    script_runner_login = g.get_user().login
    logging.info('Script is running as (authenticated user): %s.', script_runner_login)

    recent_haikus_of_the_day = fetch_recent_haikus(count=50)
    
    haikus, haikus_rated = get_rated_haikus(recent_haikus_of_the_day)
    haikus_rated_list = [json.loads(haiku_rating) for haiku_rating in haikus_rated]
    haikus_rated_avgs, haikus_rated_stds = aggregate_scores(haikus_rated_list)
    winning_haiku_idx = max(haikus_rated_avgs, key=haikus_rated_avgs.get)
    haiku_of_the_day = haikus[winning_haiku_idx]

    add_text_to_gif(
        text=haiku_of_the_day,
        gif_path=config['paths']['gif'],
        output_path=config['paths']['haiku_gif'],
        position=(50, 30),
        font_size=40,
        font_color_rgba=(240, 240, 240, 200),
    )

    update_readme(haikus, haikus_rated_list, haikus_rated_avgs, haikus_rated_stds, winning_haiku_idx)

    update_history(haikus, haikus_rated_list, haikus_rated_avgs, haikus_rated_stds, winning_haiku_idx)

    logging.info("Script finished.")
