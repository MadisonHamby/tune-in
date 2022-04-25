# spotify web api
# https://developer.spotify.com/console/playlists/ 

import requests
SPOTIFY_CREATE_PLAYLIST_URL = 'https://api.spotify.com/v1/users/314zfitamagoogjefsnhvljfcgtq/playlists'
ACCESS_TOKEN = 'BQBB_sCOB3XrWczmfKdT9ZjoT55VCO5CqK_pDZkQtZ6X_79k0RRrLr9jA8uluSEkbG79qQO2a8ZtzGbFtE3nWwxtZ_i9chK9Nkx75kDmf-jJIxVgzYrwzObLe6GCkrb4Pm50K59ltj5eAebiLZ9jhlWKYXDH6mqGkLE1-CH3M3pvTPVh30OKnk5d-iwfvi-xekec6CvVDkkFfJleL8-caMv4vS-G33Il'

# note that the access token will expire an hour or so after use
# need to get new access token

def create_playlist_on_spotify(name, public):
    response = requests.post(
        SPOTIFY_CREATE_PLAYLIST_URL,
        headers={
            "Authorization": f"Bearer {ACCESS_TOKEN}"
        },
        json={
            "name": name,
            "public" : public
      }
    )
    json_resp = response.json()
    return json_resp

def main():
    playlist = create_playlist_on_spotify(
        name="My Private Playlist",
        public=False
    )

    print(f"Playlist: {playlist}")

if __name__ == '__main__':
    main()