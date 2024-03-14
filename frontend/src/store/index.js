import { createStore } from 'vuex'
import { log } from './log/logged.js'
import { result } from './gec_result/result.js'
import { submit } from './submit/submit.js'
import { record } from './submit/submit_record.js'

export default createStore({
    modules: {
        log,
        result,
        submit,
        record,
    }
})