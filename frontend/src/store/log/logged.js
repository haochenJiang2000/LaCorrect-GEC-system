import SessionUtils from '../SessionUtils'

const state = {
    LoggedIn: SessionUtils.getValue('LoggedIn') || false,
    UserName:SessionUtils.getValue('UserName'),
    Identity:SessionUtils.getValue('Identity'),
    Privacy:SessionUtils.getValue('Privacy'),
    GetInfo:SessionUtils.getValue('GetInfo') || false,
    ruleForm:{
        UserName:SessionUtils.getValue('UserName'),
        Identity:SessionUtils.getValue('Identity'),
        Privacy:SessionUtils.getValue('Privacy'),
    }
  }

const mutations = {
    UPDATE_LOG: (state, LoggedIn) => {
        state.LoggedIn = LoggedIn
        SessionUtils.setValue('LoggedIn', LoggedIn)
    },
    UPDATE_EXIT: (state, LoggedIn) => {
        SessionUtils.cleanValue('LoggedIn')
    },
    UPDATE_GetInfo: (state, GetInfo) => {
        SessionUtils.setValue('GetInfo', GetInfo)
    },
    UPDATE_UserInfo: (state, UserName) => {
        state.UserName = UserName
        SessionUtils.setValue('UserName', UserName)
    },
    UPDATE_FormInfo: (state, ruleForm) => {
        // state.UserName = ruleForm.UserName
        // SessionUtils.setValue('UserName', ruleForm.UserName)
        state.Identity = ruleForm.Identity
        SessionUtils.setValue('Identity', ruleForm.Identity)
        state.Privacy = ruleForm.Privacy
        SessionUtils.setValue('Privacy', ruleForm.Privacy)
        state.ruleForm = {
            UserName:SessionUtils.getValue('UserName'),
            Identity:SessionUtils.getValue('Identity'),
            Privacy:SessionUtils.getValue('Privacy'),
        }
    },
}

export const log = {
    namespaced: true,
    state,
    mutations
}