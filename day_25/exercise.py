{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "6ef48226",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b03bdea1",
   "metadata": {},
   "outputs": [],
   "source": [
    "df=pd.read_csv(\"https://raw.githubusercontent.com/arewadataScience/30-Days-of-Python/main/data/hacker_news.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "cae17b83",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>title</th>\n",
       "      <th>url</th>\n",
       "      <th>num_points</th>\n",
       "      <th>num_comments</th>\n",
       "      <th>author</th>\n",
       "      <th>created_at</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>12224879</td>\n",
       "      <td>Interactive Dynamic Video</td>\n",
       "      <td>http://www.interactivedynamicvideo.com/</td>\n",
       "      <td>386</td>\n",
       "      <td>52</td>\n",
       "      <td>ne0phyte</td>\n",
       "      <td>8/4/2016 11:52</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>11964716</td>\n",
       "      <td>Florida DJs May Face Felony for April Fools' W...</td>\n",
       "      <td>http://www.thewire.com/entertainment/2013/04/f...</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "      <td>vezycash</td>\n",
       "      <td>6/23/2016 22:20</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>11919867</td>\n",
       "      <td>Technology ventures: From Idea to Enterprise</td>\n",
       "      <td>https://www.amazon.com/Technology-Ventures-Ent...</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>hswarna</td>\n",
       "      <td>6/17/2016 0:01</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>10301696</td>\n",
       "      <td>Note by Note: The Making of Steinway L1037 (2007)</td>\n",
       "      <td>http://www.nytimes.com/2007/11/07/movies/07ste...</td>\n",
       "      <td>8</td>\n",
       "      <td>2</td>\n",
       "      <td>walterbell</td>\n",
       "      <td>9/30/2015 4:12</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>10482257</td>\n",
       "      <td>Title II kills investment? Comcast and other I...</td>\n",
       "      <td>http://arstechnica.com/business/2015/10/comcas...</td>\n",
       "      <td>53</td>\n",
       "      <td>22</td>\n",
       "      <td>Deinos</td>\n",
       "      <td>10/31/2015 9:48</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "         id                                              title  \\\n",
       "0  12224879                          Interactive Dynamic Video   \n",
       "1  11964716  Florida DJs May Face Felony for April Fools' W...   \n",
       "2  11919867       Technology ventures: From Idea to Enterprise   \n",
       "3  10301696  Note by Note: The Making of Steinway L1037 (2007)   \n",
       "4  10482257  Title II kills investment? Comcast and other I...   \n",
       "\n",
       "                                                 url  num_points  \\\n",
       "0            http://www.interactivedynamicvideo.com/         386   \n",
       "1  http://www.thewire.com/entertainment/2013/04/f...           2   \n",
       "2  https://www.amazon.com/Technology-Ventures-Ent...           3   \n",
       "3  http://www.nytimes.com/2007/11/07/movies/07ste...           8   \n",
       "4  http://arstechnica.com/business/2015/10/comcas...          53   \n",
       "\n",
       "   num_comments      author       created_at  \n",
       "0            52    ne0phyte   8/4/2016 11:52  \n",
       "1             1    vezycash  6/23/2016 22:20  \n",
       "2             1     hswarna   6/17/2016 0:01  \n",
       "3             2  walterbell   9/30/2015 4:12  \n",
       "4            22      Deinos  10/31/2015 9:48  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "fba241c2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>title</th>\n",
       "      <th>url</th>\n",
       "      <th>num_points</th>\n",
       "      <th>num_comments</th>\n",
       "      <th>author</th>\n",
       "      <th>created_at</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>20094</th>\n",
       "      <td>12379592</td>\n",
       "      <td>How Purism Avoids Intels Active Management Tec...</td>\n",
       "      <td>https://puri.sm/philosophy/how-purism-avoids-i...</td>\n",
       "      <td>10</td>\n",
       "      <td>6</td>\n",
       "      <td>AdmiralAsshat</td>\n",
       "      <td>8/29/2016 2:22</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20095</th>\n",
       "      <td>10339284</td>\n",
       "      <td>YC Application Translated and Broken Down</td>\n",
       "      <td>https://medium.com/@zreitano/the-yc-applicatio...</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "      <td>zreitano</td>\n",
       "      <td>10/6/2015 14:57</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20096</th>\n",
       "      <td>10824382</td>\n",
       "      <td>Microkernels are slow and Elvis didn't do no d...</td>\n",
       "      <td>http://blog.darknedgy.net/technology/2016/01/0...</td>\n",
       "      <td>169</td>\n",
       "      <td>132</td>\n",
       "      <td>vezzy-fnord</td>\n",
       "      <td>1/2/2016 0:49</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20097</th>\n",
       "      <td>10739875</td>\n",
       "      <td>How Product Hunt really works</td>\n",
       "      <td>https://medium.com/@benjiwheeler/how-product-h...</td>\n",
       "      <td>695</td>\n",
       "      <td>222</td>\n",
       "      <td>brw12</td>\n",
       "      <td>12/15/2015 19:32</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20098</th>\n",
       "      <td>11680777</td>\n",
       "      <td>RoboBrowser: Your friendly neighborhood web sc...</td>\n",
       "      <td>https://github.com/jmcarp/robobrowser</td>\n",
       "      <td>182</td>\n",
       "      <td>58</td>\n",
       "      <td>pmoriarty</td>\n",
       "      <td>5/12/2016 1:43</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "             id                                              title  \\\n",
       "20094  12379592  How Purism Avoids Intels Active Management Tec...   \n",
       "20095  10339284          YC Application Translated and Broken Down   \n",
       "20096  10824382  Microkernels are slow and Elvis didn't do no d...   \n",
       "20097  10739875                      How Product Hunt really works   \n",
       "20098  11680777  RoboBrowser: Your friendly neighborhood web sc...   \n",
       "\n",
       "                                                     url  num_points  \\\n",
       "20094  https://puri.sm/philosophy/how-purism-avoids-i...          10   \n",
       "20095  https://medium.com/@zreitano/the-yc-applicatio...           4   \n",
       "20096  http://blog.darknedgy.net/technology/2016/01/0...         169   \n",
       "20097  https://medium.com/@benjiwheeler/how-product-h...         695   \n",
       "20098              https://github.com/jmcarp/robobrowser         182   \n",
       "\n",
       "       num_comments         author        created_at  \n",
       "20094             6  AdmiralAsshat    8/29/2016 2:22  \n",
       "20095             1       zreitano   10/6/2015 14:57  \n",
       "20096           132    vezzy-fnord     1/2/2016 0:49  \n",
       "20097           222          brw12  12/15/2015 19:32  \n",
       "20098            58      pmoriarty    5/12/2016 1:43  "
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.tail()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "d5733884",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['id', 'title', 'url', 'num_points', 'num_comments', 'author',\n",
       "       'created_at'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "05c8d251",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0                                Interactive Dynamic Video\n",
      "1        Florida DJs May Face Felony for April Fools' W...\n",
      "2             Technology ventures: From Idea to Enterprise\n",
      "3        Note by Note: The Making of Steinway L1037 (2007)\n",
      "4        Title II kills investment? Comcast and other I...\n",
      "                               ...                        \n",
      "20094    How Purism Avoids Intels Active Management Tec...\n",
      "20095            YC Application Translated and Broken Down\n",
      "20096    Microkernels are slow and Elvis didn't do no d...\n",
      "20097                        How Product Hunt really works\n",
      "20098    RoboBrowser: Your friendly neighborhood web sc...\n",
      "Name: title, Length: 20099, dtype: object\n"
     ]
    }
   ],
   "source": [
    "title_series = df['title']\n",
    "print(title_series)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "d3ac388b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(20099, 7)"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "698e5140",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             id                                              title  \\\n",
      "102    10974870                From Python to Lua: Why We Switched   \n",
      "103    11244541          Ubuntu 16.04 LTS to Ship Without Python 2   \n",
      "144    10963528  Create a GUI Application Using Qt and Python i...   \n",
      "196    10716331  How I Solved GCHQ's Xmas Card with Python and ...   \n",
      "436    11895088  Unikernel Power Comes to Java, Node.js, Go, an...   \n",
      "...         ...                                                ...   \n",
      "19597  12061177  David Beazley  Python Concurrency from the Gro...   \n",
      "19852  10988468    Ask HN: How to automate Python apps deployment?   \n",
      "19862  11738470                          Moving Away from Python 2   \n",
      "19980  12524656                      Python vs. Julia Observations   \n",
      "19998  11735438  Show HN: Decorating: Animated pulsed for your ...   \n",
      "\n",
      "                                                     url  num_points  \\\n",
      "102    https://www.distelli.com/blog/using-lua-for-ou...         243   \n",
      "103    http://news.softpedia.com/news/ubuntu-16-04-lt...           2   \n",
      "144                        http://digitalpeer.com/s/c63e          21   \n",
      "196    http://matthewearl.github.io/2015/12/10/gchq-x...           6   \n",
      "436    http://www.infoworld.com/article/3082051/open-...           3   \n",
      "...                                                  ...         ...   \n",
      "19597        https://www.youtube.com/watch?v=MCs5OvhV9S4           2   \n",
      "19852                                                NaN           4   \n",
      "19862  https://asmeurer.github.io/blog/posts/moving-a...         227   \n",
      "19980  https://medium.com/@Jernfrost/python-vs-julia-...           2   \n",
      "19998             https://github.com/ryukinix/decorating           3   \n",
      "\n",
      "       num_comments        author        created_at  \n",
      "102             188      chase202   1/26/2016 18:17  \n",
      "103               1       _snydly    3/8/2016 10:39  \n",
      "144               1        zoodle   1/24/2016 19:01  \n",
      "196               1          kipi  12/11/2015 10:38  \n",
      "436               1  syslandscape   6/13/2016 16:23  \n",
      "...             ...           ...               ...  \n",
      "19597             1      bakery2k    7/9/2016 13:05  \n",
      "19852            18       aalhour   1/28/2016 14:55  \n",
      "19862           275     ngoldbaum   5/20/2016 15:14  \n",
      "19980             1   blacksmythe    9/18/2016 9:54  \n",
      "19998             1         lerax    5/20/2016 3:48  \n",
      "\n",
      "[160 rows x 7 columns]\n"
     ]
    }
   ],
   "source": [
    "python_titles = df[df['title'].str.contains('python', case=False)]\n",
    "print(python_titles)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "7679926d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "             id                                              title  \\\n",
      "267    12352636   Show HN: Hire JavaScript - Top JavaScript Talent   \n",
      "580    10871330  Python integration for the Duktape Javascript ...   \n",
      "811    10741251  Ask HN: Are there any projects or compilers wh...   \n",
      "1046   11343334  If you write JavaScript tools or libraries, bu...   \n",
      "1093   10422726  Rollup.js: A next-generation JavaScript module...   \n",
      "...         ...                                                ...   \n",
      "19349  11448301    Fotorama, a responsive JavaScript photo gallery   \n",
      "19548  12105148                 Another Kind of JavaScript Fatigue   \n",
      "19610  12203508  Lonely programmer detective uncovers the Mozil...   \n",
      "19885  12552131  Ask HN: Best Practices for CSS in a Modern Jav...   \n",
      "20069  12149183  Show HN: Parse recipe ingredients using JavaSc...   \n",
      "\n",
      "                                                     url  num_points  \\\n",
      "267                              https://www.hirejs.com/           1   \n",
      "580               https://pypi.python.org/pypi/pyduktape           3   \n",
      "811                                                  NaN           1   \n",
      "1046   https://medium.com/@Rich_Harris/how-to-not-bre...          48   \n",
      "1093                                 http://rollupjs.org          57   \n",
      "...                                                  ...         ...   \n",
      "19349                                http://fotorama.io/           1   \n",
      "19548  http://chrismm.com/blog/the-other-kind-of-java...           9   \n",
      "19610         http://stackoverflow.com/a/38677222/984780          29   \n",
      "19885                                                NaN           6   \n",
      "20069       https://github.com/herkyl/ingredients-parser           6   \n",
      "\n",
      "       num_comments          author        created_at  \n",
      "267               1        eibrahim   8/24/2016 15:16  \n",
      "580               1         stefano    1/9/2016 14:26  \n",
      "811               2         ggonweb  12/15/2015 23:26  \n",
      "1046             19     callumlocke   3/23/2016 10:54  \n",
      "1093             17         dmmalam   10/21/2015 0:02  \n",
      "...             ...             ...               ...  \n",
      "19349             1         alexkon    4/7/2016 15:59  \n",
      "19548             2    JacksCracked    7/16/2016 3:44  \n",
      "19610             8    luisperezphd    8/1/2016 16:07  \n",
      "19885             6        xwvvvvwx   9/21/2016 20:53  \n",
      "20069             2  zongitsrinzler   7/23/2016 11:35  \n",
      "\n",
      "[170 rows x 7 columns]\n"
     ]
    }
   ],
   "source": [
    "js_titles = df[df['title'].str.contains('javascript', case=False)]\n",
    "print(js_titles)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f8e7a82f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 20099 entries, 0 to 20098\n",
      "Data columns (total 7 columns):\n",
      " #   Column        Non-Null Count  Dtype \n",
      "---  ------        --------------  ----- \n",
      " 0   id            20099 non-null  int64 \n",
      " 1   title         20099 non-null  object\n",
      " 2   url           17659 non-null  object\n",
      " 3   num_points    20099 non-null  int64 \n",
      " 4   num_comments  20099 non-null  int64 \n",
      " 5   author        20099 non-null  object\n",
      " 6   created_at    20099 non-null  object\n",
      "dtypes: int64(3), object(4)\n",
      "memory usage: 1.1+ MB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "fd9086b3",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>num_points</th>\n",
       "      <th>num_comments</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>2.009900e+04</td>\n",
       "      <td>20099.000000</td>\n",
       "      <td>20099.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>mean</th>\n",
       "      <td>1.131755e+07</td>\n",
       "      <td>50.296632</td>\n",
       "      <td>24.803025</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>std</th>\n",
       "      <td>6.964531e+05</td>\n",
       "      <td>107.110322</td>\n",
       "      <td>56.108639</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>min</th>\n",
       "      <td>1.017691e+07</td>\n",
       "      <td>1.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25%</th>\n",
       "      <td>1.070172e+07</td>\n",
       "      <td>3.000000</td>\n",
       "      <td>1.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>50%</th>\n",
       "      <td>1.128452e+07</td>\n",
       "      <td>9.000000</td>\n",
       "      <td>3.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>75%</th>\n",
       "      <td>1.192613e+07</td>\n",
       "      <td>54.000000</td>\n",
       "      <td>21.000000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>max</th>\n",
       "      <td>1.257898e+07</td>\n",
       "      <td>2553.000000</td>\n",
       "      <td>1733.000000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                 id    num_points  num_comments\n",
       "count  2.009900e+04  20099.000000  20099.000000\n",
       "mean   1.131755e+07     50.296632     24.803025\n",
       "std    6.964531e+05    107.110322     56.108639\n",
       "min    1.017691e+07      1.000000      1.000000\n",
       "25%    1.070172e+07      3.000000      1.000000\n",
       "50%    1.128452e+07      9.000000      3.000000\n",
       "75%    1.192613e+07     54.000000     21.000000\n",
       "max    1.257898e+07   2553.000000   1733.000000"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "39d99a18",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "id                 0\n",
       "title              0\n",
       "url             2440\n",
       "num_points         0\n",
       "num_comments       0\n",
       "author             0\n",
       "created_at         0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 12,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "c612cd57",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "ingve         198\n",
       "prostoalex    123\n",
       "dnetesn        98\n",
       "jseliger       85\n",
       "jonbaer        85\n",
       "             ... \n",
       "grobie          1\n",
       "magoghm         1\n",
       "VinceD01        1\n",
       "chrtze          1\n",
       "brw12           1\n",
       "Name: author, Length: 10382, dtype: int64"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.author.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "ca2edc5d",
   "metadata": {},
   "outputs": [],
   "source": [
    "d"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
