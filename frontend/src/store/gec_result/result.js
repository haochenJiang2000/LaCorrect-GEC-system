import SessionUtils from '../SessionUtils'

const state = {
    Date: SessionUtils.getValue('Date'),
    FileName: SessionUtils.getValue('FileName'),
    Errors: SessionUtils.getValue('Errors'),
    Id: SessionUtils.getValue('id') || false,
  }

const mutations = {
    UPDATE_GECRESULT: (state, resp) => {
        state.Date = resp[0]
        SessionUtils.setValue('Date', resp[0])
        state.FileName = resp[1]
        SessionUtils.setValue('FileName', resp[1])
        state.Errors = resp[2]
        SessionUtils.setValue('Errors', resp[2])
    },
    UPDATE_JUDGE: (state, [Errors, idx, judge]) => {
        state.Errors[idx].judge = judge
        SessionUtils.setValueDeep('Errors', Errors, idx, judge)
    },
    UPDATE_FileId: (state, Id) => {
        state.Id = Id
        SessionUtils.setValue('id', Id)
    },
}
  
export const result = {
    namespaced: true,
    state,
    mutations
}