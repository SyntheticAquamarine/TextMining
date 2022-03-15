import re

prw = "Dzisiaj mamy 4 stopnie na plusie, 1 marca 2022 roku"
res_1 = re.sub('\d+','',prw)

scnd = '<div><h2>Header</h2> <p>article<b>strong text</b> <a href="">link</a></p></div>'
res_2 = re.sub('<.{1,9}>','',scnd)

trd = "Lorem ipsum dolor sit amet, consectetur; adipiscing elit. \
Sed eget mattis sem. Mauris egestas erat quam, ut faucibus eros \
congue et. In blandit, mi eu porta; lobortis, tortor nisl facilisis leo,\
at tristique augue risus eu risus."

res_3 = re.sub('(?:[";.,:...?!-])','',trd)

ex2 = " Lorem ipsum dolor sit amet, consectetur adipiscing elit.\
Sed #texting eget mattis sem. Mauris #frasista egestas erat #tweetext quam,\
ut faucibus eros #frasier congue et. In blandit, mi eu porta lobortis,\
tortor nisl facilisis leo, at tristique #frasistas augue risus eu risus."

res_ex2 = re.findall('#\w*',ex2)

ex3 = "sit amet, ;< consectetur :> adipiscing :-) elit. Sed #texting eget :) mattis sem. Mauris #frasista\
egestas erat #tweetext ;( quam, ut :< faucibus eros #frasier congue et. In blandit, mi eu porta\
lobortis, :> tortor nisl facilisis leo, ;) at tristique #frasistas :) augue risus eu risus."

res_ex3 = re.findall('[:;]+-?[>)(<]',ex3)


