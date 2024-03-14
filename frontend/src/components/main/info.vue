<template>
  <div v-if="LoggedIn">
    <el-card class="box-card" style="width: 1000px;">
      <template #header>
        <div class="card-header">
          <span>用户信息</span>
          <!-- <el-button class="button" text>Operation button</el-button> -->
        </div>
      </template>
      
      <el-form
        ref="ruleFormRef"
        :model="ruleForm"
        :rules="rules"
        label-width="120px"
        class="demo-ruleForm"
        :size="formSize"
        status-icon
      >
        <el-form-item label="用户名" prop="name">
          <el-col :span="7">
            <el-input disabled v-model="ruleForm.UserName" />
          </el-col>
        </el-form-item>
        <!-- <el-form-item label="密码" prop="password">
          <el-input v-model="ruleForm.PassWord" 
          type="password" />
        </el-form-item> -->
        <el-form-item label="身份" prop="identity">
          <el-select v-model="ruleForm.Identity" placeholder="航天研究院用户">
            <el-option label="航天研究院用户" value="航天研究院用户" />
          </el-select>
        </el-form-item>
        <!-- <el-form-item label="隐私设置" prop="privacy">
          <el-select v-model="ruleForm.Privacy" placeholder="可公开使用">
            <el-option label="可公开使用" value="可公开使用" />
            <el-option label="乐改团队内部使用" value="乐改团队内部使用" />
            <el-option label="不可使用" value="不可使用" />
          </el-select>
        </el-form-item> -->
        <el-form-item>
          <el-button type="primary" @click="submitForm()">
            提交修改
          </el-button>
          <el-button type="default" @click="dlg_password = true">修改密码</el-button>
        </el-form-item>
      </el-form>
      <el-dialog
        v-model="dlg_password"
        title="密码修改"
        width="600px"
        destroy-on-close
        align-center
      >
        <el-form ref="form" :model="form" label-width="80px">
          <el-form-item label="原密码">
            <el-col :span="15">
              <el-input placeholder="请输入原密码" 
              v-model="form.PassWord"
              type="password"
              ></el-input>
            </el-col>
          </el-form-item>
          <el-form-item label="新密码">
            <el-col :span="15">
              <el-input placeholder="请输入新密码" 
              v-model="form.newPassWord1"
              type="password"
              ></el-input>
            </el-col>
          </el-form-item>
          <el-form-item label="新密码">
            <el-col :span="15">
              <el-input placeholder="请再次输入新密码" 
              v-model="form.newPassWord2"
              type="password"
              ></el-input>
            </el-col>
          </el-form-item>
          <el-form-item>
            <el-col :span="5">
              <el-button type="primary" @click="ChangePassWord()">确认修改</el-button>
            </el-col>
            <el-col :span="5" style="font-size: 10px;">
              <el-button link size="small" type="default" @click.prevent="dlg_password = false">取消</el-button>
            </el-col>
          </el-form-item>
        </el-form>
      </el-dialog>
    </el-card>
  </div>
  <div v-else>
    <el-space wrap>
      <el-card class="box-card" style="width: 1000px">
        <template #header>
          <div class="card-header">
            请先登录
          </div>
        </template>
        <div class="text item">
          <h1>提交记录</h1>
        </div>
      </el-card>
    </el-space>  
  </div>
</template>
  
<script lang="ts" setup>
  import { reactive, ref } from 'vue'
  import type { FormInstance, FormRules } from 'element-plus'
  import axios from 'axios';
  import { computed, watch } from 'vue'
  import { useStore } from 'vuex'
  import { ElMessage, ElMessageBox } from 'element-plus'
  import type { Action } from 'element-plus'
  import { nextTick } from 'vue'

  const store = useStore()
  const LoggedIn = computed(() => store.state.log.LoggedIn || false)

  const formSize = ref('default')
  const ruleFormRef = ref<FormInstance>()

  const dlg_password = ref(false)
  const dlg_forget = ref(false)
  // const PassWord = ref("")
  // const newPassWord1 = ref("")
  // const newPassWord2 = ref("")

  const form = reactive({
    PassWord: "",
    newPassWord1: "",
    newPassWord2: "",
  })

  // 后端请求用户信息
  watch(LoggedIn, () => {
    let formData = new FormData();
    formData.append("UserName", store.state.log.UserName);
    console.log("请求用户信息...")
    axios.post('/information', formData).then(res => {
      store.commit('log/UPDATE_FormInfo', res.data)
      console.log("请求用户信息成功")
      store.commit('log/UPDATE_GetInfo', true)
    })
  })
  
  let formData = new FormData();
  formData.append("UserName", store.state.log.UserName);
  axios.post('/information', formData).then(res => {
    store.commit('log/UPDATE_FormInfo', res.data)
    console.log("请求记录成功")
    store.commit('log/UPDATE_GetInfo', true)
  })
  const ruleForm = computed(() => store.state.log.ruleForm)

  const rules = reactive<FormRules>({
    username: [
      { required: false, message: '请输入邮箱(仅限学校认证的邮箱)', trigger: 'blur' },
      { min: 6, max: 30, message: '请输入正确的邮箱地址', trigger: 'blur' },
    ],
    privacy: [
      {
        required: false,
        message: '请设置隐私等级',
        trigger: 'change',
      },
    ],
    identity: [
      {
        required: false,
        message: '请选择你的身份',
        trigger: 'change',
      },
    ],
  })
  
  const submitForm = () => {
    let formData = new FormData();
    formData.append("UserName", store.state.log.ruleForm.UserName);
    formData.append("Identity", store.state.log.ruleForm.Identity);
    formData.append("Privacy", store.state.log.ruleForm.Privacy);
    axios.post('/update_information', formData).then(res => {
      if(res.data.code != 707){
        if(res.data.message=="修改成功！") {
          // store.commit('log/UPDATE_FormInfo', ruleForm)
          console.log("信息更新成功：", ruleForm)
          ElMessageBox.alert('信息更新成功!', '提示', {
            confirmButtonText: 'OK',
            callback: (action: Action) => {
              ElMessage({
                type: 'info',
                message: `信息更新成功!`,
              })
            },
          })
        }
      }
      if (res.data.message) {
        ElMessage(res.data.message)
      }
    })
  }

  const ChangePassWord = () => {
    if(form.newPassWord1 != form.newPassWord2) {
      ElMessage("两次输入的新密码不一致！")
    }
    else{
      let formData = new FormData();
      console.log(store.state.log.ruleForm.UserName)
      console.log(form.PassWord)
      console.log(form.newPassWord1)
      formData.append("UserName", store.state.log.ruleForm.UserName);
      formData.append("PassWord", form.PassWord);
      formData.append("newPassWord", form.newPassWord1);
      axios.post('/update_password', formData).then(res => {
        console.log(res.data)
        if(res.data.code != 707){
          if(res.data) {
            ElMessageBox.alert(res.data.message, '提示', {
              confirmButtonText: 'OK',
              callback: (action: Action) => {
                ElMessage({
                  type: 'info',
                  message: res.data.message,
                })
              },
            })
          }
          else {
            ElMessage("未知错误")
          }
        }
        else{
          store.commit('log/UPDATE_LOG', false)
          store.commit('log/UPDATE_EXIT', false)
        }
        if (res.data.message) {
          ElMessage(res.data.message)
        }
      })
    }
    dlg_password.value = false
  }
</script>