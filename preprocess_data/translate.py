import json
import time

from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.tmt.v20180321 import tmt_client, models


class Translator():
    def __init__(self):
        cred = credential.Credential("AKIDxfARbVDcnkSnvqBx86pID4h3DBKGPNWb", "5tTntiGXUEhtpWyyB8qIX6OUXJhqiOoy")
        httpProfile = HttpProfile()
        httpProfile.endpoint = "tmt.tencentcloudapi.com"
        httpProfile.reqTimeout = 30

        clientProfile = ClientProfile()
        clientProfile.httpProfile = httpProfile
        self.client = tmt_client.TmtClient(cred, "ap-shanghai", clientProfile)


    def __do_translate__(self, sentence, source, target):
        req = models.TextTranslateRequest()
        params = {
            "SourceText": sentence,
            "Source": source,
            "Target": target,
            "ProjectId": 0
        }
        req.from_json_string(json.dumps(params))
        resp = self.client.TextTranslate(req)
        return resp.TargetText

    def __call__(self, doc):
        middle_languages = ['zh', 'ja', 'ko', 'fr', 'es', 'it', 'de', 'tr', 'ru', 'pt']
        ans = []
        sentences = doc.split('.')
        print('split sentences:{}'.format(len(sentences)))
        for cur_middle in middle_languages:
            doc_cur_middle = []
            for one_s in sentences:
                if (len(one_s) == 0):
                    continue
                # print(one_s)
                midde_text = self.__do_translate__(one_s, 'en', cur_middle)
                # print(midde_text)
                time.sleep(0.3)
                res = self.__do_translate__(midde_text, cur_middle, 'en')
                # print(res)
                doc_cur_middle.append(res)
                time.sleep(0.3)
            s_doc_cur_middle = ' '.join(doc_cur_middle) + '.'
            # print(s_doc_cur_middle)
            # print('--------')
            ans.append(s_doc_cur_middle)
        return ans

    def translate_sentence_whole(self,sentence):
        middle_languages = ['zh', 'ja', 'ko', 'fr', 'es', 'it', 'de', 'tr', 'ru', 'pt']
        ans = []
        for cur_middle in middle_languages:
            print(cur_middle)
            midde_text = self.__do_translate__(sentence, 'en', cur_middle)
            time.sleep(0.3)
            res = self.__do_translate__(midde_text, cur_middle, 'en')
            time.sleep(0.3)
            ans.append(res)
        return ans




if __name__ == '__main__':
    translator = Translator()
    res = translator(
        'Thousands are taking part in a third day of street protests. An Australian economic adviser to Ms Suu Kyi, Sean Turnell, has also been detained and on Monday his family posted a statement on Facebook calling for his immediate release.')
    print(res)
