import tweepy, re, requests
from secrets import *
import tests as t
from bs4 import BeautifulSoup
import code, random, traceback

auth = tweepy.OAuthHandler(C_KEY, C_SECRET)
auth.set_access_token(A_TOKEN, A_TOKEN_SECRET)
api = tweepy.API(auth)

def find_mentions():
    mentions = api.mentions_timeline(count=5)
    #code.interact(local=locals())
    for mention in mentions:
        print '{} said: "{}"'.format(mention.user.screen_name, mention.text)

        # parse tweet for url
        try:
            tweet_url = re.search(r'https:.*\b', mention.text).group(0)
            text = solve_tweeted_set(tweet_url)
        except AttributeError:
            # ignore tweets with no image
            text = "You didn't tweet an image ya dumbo, {}".format(mention.user.name)
            #api.update_status( text, in_reply_to_status_id = mention.id )
            n = random.randint(0, 140-len(text))
            text += '!'*n

        status = '@{} {}'.format(mention.user.screen_name, text)

        api.retweet(id=mention.id)

        send_tweet(status)

def solve_tweeted_set(tweet_url):
    tweet_html = requests.get(tweet_url).content

    # scrape tweet HTML string for image url
    soup = BeautifulSoup(tweet_html, 'lxml')
    img_url = soup.find('meta', attrs={'property': 'og:image'})['content']

    # find Sets
    num_sets, out_img = t.play_game(img_url, path_is_url=True)

    if num_sets:
        text = "Whoa! {} sets...that's cray".format(num_sets)

    else:
        text = "No sets...bummer."

    return text

def send_tweet(text):
    try:
        text = text[:140]
        api.update_status( status = text )
        #, in_reply_to_status_id = mention.id )
        #api.update_with_media()
    except:
        print 'hello'
        traceback.print_exc()
        pass
