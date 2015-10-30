import tweepy, re, requests, StringIO
from secrets import *
import tests as t
from bs4 import BeautifulSoup
import code, random, traceback
from tweepy.streaming import StreamListener
from tweepy.binder import bind_api

class myAPI(tweepy.API):
    @staticmethod
    def _pack_image(filename, max_size, form_field="image", f=None):
        """Pack image from file into multipart-formdata post body"""
        ## image must be less than 700kb in size if f is None:
            #try:
                #if os.path.getsize(filename) > (max_size * 1024):
                    #raise TweepError('File is too big, must be less than %skb.' % max_size)
            #except os.error as e:
                #raise TweepError('Unable to access file: %s' % e.strerror)
#
            ## build the mulitpart-formdata body
            #fp = open(filename, 'rb')
        #else:
            #f.seek(0, 2)  # Seek to end of file
            #if f.tell() > (max_size * 1024):
                #raise TweepError('File is too big, must be less than %skb.' % max_size)
            #f.seek(0)  # Reset to beginning of file
            #fp = f
#
        ## image must be gif, jpeg, or png
        #file_type = mimetypes.guess_type(filename)
        #if file_type is None:
            #raise TweepError('Could not determine file type')
        #file_type = file_type[0]
        #if file_type not in ['image/gif', 'image/jpeg', 'image/png']:
            #raise TweepError('Invalid file type for image: %s' % file_type)
#
        #if isinstance(filename, six.text_type):
            #filename = filename.encode("utf-8")

        BOUNDARY = b'Tw3ePy'
        body = list()
        body.append(b'--' + BOUNDARY)
        body.append('Content-Disposition: form-data; name="{0}";'
                    ' filename="{1}"'.format(form_field, filename)
                    .encode('utf-8'))


        file_type = 'image/jpeg'
        body.append('Content-Type: {0}'.format(file_type).encode('utf-8'))
        body.append(b'')

        body.append('Content-Transfer-Encoding: base64'.encode('utf-8'))
        body.append(b'')

        #body.append(fp.read())
        body.append(f.encode('utf-8'))

        body.append(b'--' + BOUNDARY + b'--')
        body.append(b'')
        #fp.close()
        body = b'\r\n'.join(body)

        # build headers
        headers = {
            'Content-Type': 'multipart/form-data; boundary=Tw3ePy',
            'Content-Length': str(len(body))
        }

        return headers, body

    #def _pack_image(filename, max_size, form_field, f):
        #"""Pack image from file into multipart-formdata post body"""
        ## image must be less than 700kb in size
        ##try:
            ##if os.path.getsize(filename) > (max_size * 1024):
                ##raise TweepError('File is too big, must be less than 700kb.')
        ##except os.error, e:
            ##raise TweepError('Unable to access file')
#
        ## image must be gif, jpeg, or png
        ##file_type = mimetypes.guess_type(filename)
        ##if file_type is None:
            ##raise TweepError('Could not determine file type')
        ##file_type = file_type[0]
        ##if file_type not in ['image/gif', 'image/jpeg', 'image/png']:
            ##raise TweepError('Invalid file type for image: %s' % file_type)
#
        ## build the multipart-formdata body
        ##fp = open(filename, 'rb')
        #BOUNDARY = 'Tw3ePy'
        #body = []
        #body.append('--' + BOUNDARY)
        #body.append('Content-Disposition: form-data; name="image"; filename="%s"' % filename)
#
        #file_type = 'image/jpeg'
#
        #body.append('Content-Type: %s' % file_type)
        #body.append('')
#
        #body.append('Content-Transfer-Encoding: base64')
        #body.append('')
#
        ##body.append(fp.read())
        #body.append(f)
        #body.append('--' + BOUNDARY + '--')
        #body.append('')
        ##fp.close()
        #body = '\r\n'.join(body)
#
        ## build headers
        #headers = {
            #'Content-Type': 'multipart/form-data; boundary=Tw3ePy',
            #'Content-Length': len(body)
        #}
#
        #return headers, body
#
    def media_upload(self, filename, *args, **kwargs):
        """ :reference: https://dev.twitter.com/rest/reference/post/media/upload
            :allowed_param:
        """
        f = kwargs.pop('file', None)
        headers, post_data = myAPI._pack_image(filename, 3072, form_field='media', f=f)
        kwargs.update({'headers': headers, 'post_data': post_data})

        #kwargs.update({'headers': headers, 'media_data': post_data})
        kwargs.update({'media_data': f})

        return bind_api(
            api=self,
            path='/media/upload.json',
            method='POST',
            payload_type='media',
            allowed_param=[],
            require_auth=True,
            upload_api=True
        )(*args, **kwargs)


auth = tweepy.OAuthHandler(C_KEY, C_SECRET)
auth.set_access_token(A_TOKEN, A_TOKEN_SECRET)
#api = myAPI(auth, timeout=600)
api = tweepy.API(auth)

class listener_tweeter(StreamListener):

    def _onconnect(self):
        print 'Connected! Listening...'

    # override on_status to pass data from on_data method of tweepy's StreamListener
    def on_status(self, status):
        print '{} said: "{}"'.format(status.user.screen_name, status.text)

        # ignore retweets or tweets from self
        if status.retweeted or ( status.user.screen_name == 'ProfessorSet' ):
            return

        img_str = None

        # parse tweet for url
        try:
            tweet_url = re.search(r'https:.*\b', status.text).group(0)
            text, img_str = solve_tweeted_set(tweet_url)

        #ignore tweets with no image
        except AttributeError:
            text = "You forgot a picture, {} #whoops #howembarrassing".format(status.user.name)
            n = random.randint(0, 140-len(text))
            text += '!'*n

        response_text = '.@{} {}'.format(status.user.screen_name, text)
        print response_text

        try:
            api.retweet(id=status.id)

        except tweepy.error.TweepError:
            traceback.print_exc()

        send_tweet( response_text, response_id=status.id, media_str=img_str )


    def on_error(self, status_code):
        print 'Error: {}'.format(status_code)

    def on_exception(self, exception):
        print 'Exception: {}'.format(exception)
        return




def listen():
    my_listener = listener_tweeter()
    stream = tweepy.Stream(auth, my_listener)
    stream.userstream(_with='user')



#def find_mentions():
    #mentions = api.mentions_timeline(count=5)
#
    #for mention in mentions[::-1]:
        #print '{} said: "{}"'.format(mention.user.screen_name, mention.text)
#
        ## parse tweet for url
        #try:
            #tweet_url = re.search(r'https:.*\b', mention.text).group(0)
            #text = solve_tweeted_set(tweet_url)
        #except AttributeError:
            ## ignore tweets with no image
            #text = "You forgot a picture, {} #whoops #howembarrassing".format(mention.user.name)
            ##api.update_status( text, in_reply_to_status_id = mention.id )
            #n = random.randint(0, 140-len(text))
            #text += '!'*n
#
        #status_text = '.@{} {}'.format(mention.user.screen_name, text)
#
        #try:
            #api.retweet(id=mention.id)
        #except tweepy.error.TweepError:
            #traceback.print_exc()
            #continue
#
        #send_tweet( status_text, response_id=mention.id )


def solve_tweeted_set(tweet_url):
    tweet_content = requests.get(tweet_url).content

    # scrape tweet HTML string for image url
    soup = BeautifulSoup(tweet_content, 'lxml')
    img_url = soup.find('meta', attrs={'property': 'og:image'})['content']

    # find Sets
    num_sets, img_str = t.play_game(img_url, path_is_url=True)#, pop_open=False)

    text = ("Whoa! {} sets #craycray".format(num_sets) if num_sets \
           else "No sets #bummer #sadface")

    img_str = (img_str if num_sets else None)

    return (text, img_str)


def send_tweet(text, response_id=None, media_str=None):
    try:
        text = text[:140]
        d_arguments = {'status': text}

        if response_id: d_arguments['in_reply_to_status_id'] = response_id

        if media_str:
            #upload = api.media_upload(file=media_str, filename='myfile')
            upload = api.media_upload(filename='tmp.jpeg')

            d_arguments['media_ids'] = [upload.media_id_string]

        api.update_status(**d_arguments)

    except:
        traceback.print_exc()
        pass


if __name__ == "__main__":
    listen()
