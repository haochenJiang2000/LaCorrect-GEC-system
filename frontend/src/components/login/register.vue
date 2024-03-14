<template>
  <form v-if="!loggedIn">
    <div v-if="!isRegister">
      <div class="rg_layout1">
        <div class="rg_left">
          <p>登录</p>
          <p>USER LOGIN</p>
        </div>
        <div class="rg_center">
          <div class="rg_form">
            <div style="margin: 50px 0;"></div>
            <el-form ref="form" :model="form" :rules="rules" label-width="80px">
              <el-form-item label="Email" prop="Email">
                <el-col :span="15">
                  <el-input placeholder="请输入邮箱（edu.cn）" v-model="form.Email"></el-input>
                </el-col>
              </el-form-item>
              <el-form-item label="密码">
                <el-col :span="15">
                  <el-input placeholder="请输入密码" 
                  v-model="password"
                  type="password"
                  ></el-input>
                </el-col>
              </el-form-item>
              <el-form-item>
                <el-col :span="5">
                  <el-button type="primary" @click="login">登录</el-button>
                </el-col>
                <el-col :span="5" style="font-size: 10px;">
                  <el-button link size="small" type="default" @click.prevent="change">注册</el-button>
                </el-col>
                <el-col :span="5" style="font-size: 10px;">
                  <el-button type="default" size="small" text @click="dlg_forget = true">忘记密码</el-button>
                </el-col>
              </el-form-item>
            </el-form>
          </div>
        </div>
      </div>
    </div>
    <div v-else>
      <div class="rg_layout2">
        <div class="rg_left">
          <p>新用户注册</p>
          <p>USER REGISTER</p>
        </div>
        <div class="rg_center">
          <div class="rg_form">
            <div style="margin: 50px 0;"></div>
            <el-form ref="form" :model="form" :rules="rules" label-width="80px">
              <el-form-item label="Email" prop="Email">
                <el-col :span="15">
                  <el-input placeholder="请输入邮箱（edu.cn）" v-model="form.Email"></el-input>
                </el-col>
                <el-col :span="9">
                  <el-button type="success" plain @click="sendEmail()">发送邮件验证</el-button>
                </el-col>
              </el-form-item>
              <el-form-item label="密码">
                <el-col :span="15">
                  <el-input placeholder="请输入密码" 
                  v-model="password"
                  type="password"
                  ></el-input>
                </el-col>
              </el-form-item>
              <el-form-item label="验证码">
                <el-col :span="10">
                  <el-input
                      type="text"
                      placeholder=""
                      v-model="form.text"
                      oninput="value=value.replace(/\D/g,'')"
                      maxlength="6"
                      show-word-limit
                  >
                  </el-input>
                </el-col>
              </el-form-item>
              <!-- <el-form-item>
                <div class="mb-2 flex items-center text-sm">
                  <div>
                    <el-checkbox v-model="checked" label="用户隐私保护声明" size="large" @change="Info"/>
                  </div>
                </div>
              </el-form-item> -->
              <el-form-item>
                <el-col :span="5">
                  <el-button v-if="!checked" type="primary" @click="register" disabled >注册</el-button>
                  <el-button v-if="checked" type="primary" @click="register">注册</el-button>
                </el-col>
                <el-col :span="10" style="font-size: 10px;">
                  <el-button link size="small" type="default" @click.prevent="change">登录</el-button>
                </el-col>
              </el-form-item>
            </el-form>
          </div>
        </div>
      </div>
    </div>
    <!-- <el-dialog
      v-model="dialog"
      width="1000px" 
      @close="close"
      center
    >
      <div>
        <Privacy></Privacy>
      </div>
      <div>
          &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
          &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
          &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
          &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
          &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
          &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
          &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
          &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
          &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
          &nbsp&nbsp&nbsp&nbsp&nbsp&nbsp
          <el-button type="primary" class="button" @click="close()">我已阅读并了解了乐改隐私政策</el-button>
      </div>
    </el-dialog> -->
    <el-dialog
        v-model="dlg_forget"
        title="密码修改"
        width="600px"
        destroy-on-close
        align-center
      >
        <el-form ref="form" :model="pswd_form" label-width="80px">
          <el-form-item label="邮箱认证">
            <el-col :span="15">
              <el-input placeholder="学校认证的邮箱（edu.cn）" v-model="pswd_form.Email"></el-input>
            </el-col>
            <el-col :span="9">
              <el-button type="success" plain @click="sendEmail2()">发送邮件验证</el-button>
            </el-col>
          </el-form-item>
          <el-form-item label="新密码">
            <el-col :span="15">
              <el-input placeholder="请输入新密码" 
              v-model="pswd_form.newPassWord1"
              type="password"
              ></el-input>
            </el-col>
          </el-form-item>
          <el-form-item label="新密码">
            <el-col :span="15">
              <el-input placeholder="请再次输入新密码" 
              v-model="pswd_form.newPassWord2"
              type="password"
              ></el-input>
            </el-col>
          </el-form-item>
          <el-form-item label="验证码">
            <el-col :span="10">
              <el-input
                  type="text"
                  placeholder=""
                  v-model="pswd_form.text"
                  oninput="value=value.replace(/\D/g,'')"
                  maxlength="6"
                  show-word-limit
              >
              </el-input>
            </el-col>
          </el-form-item>
          <el-form-item>
            <el-col :span="5">
              <el-button type="primary" @click="ChangePassWord()">确认修改</el-button>
            </el-col>
            <el-col :span="5" style="font-size: 10px;">
              <el-button link size="small" type="default" @click.prevent="dlg_forget = false">取消</el-button>
            </el-col>
          </el-form-item>
        </el-form>
      </el-dialog>
  </form>
  <div v-else>
    <h3>欢迎 {{username}}</h3>
    <el-button type="primary" @click="exit">关闭</el-button>
  </div>
</template>
  
  <script>
  import { ref,h,computed, getCurrentInstance,reactive } from 'vue'
  import { createLogger, useStore } from 'vuex'
  import axios from 'axios'
  import { ElMessage, ElMessageBox } from 'element-plus'
  import Privacy from './privacy.vue'
  // import { upload_url } from '../../../public/config.json'
  // import JSEncrypt from 'jsencrypt/bin/jsencrypt'

  let xmlhttp = new XMLHttpRequest();
  xmlhttp.open('get', './config.json', false);
  xmlhttp.send(null);
  let url = JSON.parse(xmlhttp.responseText).url
  let upload_url = JSON.parse(xmlhttp.responseText).upload_url

  export default {
    mounted() {
      this.$store.state.yesOrNo = false
    },
    name: "signUp",
    components: {
      Privacy,
    },
    data: function () {
      return {
        isRegister: false,
        loggedIn: false,
        store: useStore(),
        proxy: getCurrentInstance(),
        checked: ref(true),
        dialog: ref(false),
        dlg_forget: ref(false),
        password: "",


        // encrypt: new JSEncrypt(),
        // pubKey: '*****',

        form: reactive({
          Email: '',
          password: "",
          identity: "航天研究院用户",
          text: '',
          radio: '1',
          date: '',
        }),
        pswd_form: reactive({
          Email: '',
          newPassWord1: "",
          newPassWord2: "",
          text: "",
        }),
        rules: {
          Email: [{required: false, message: '请使用学校认证的邮箱(edu.cn)', trigger: 'blur'}],
          password: [{required: false, message: '密码需要大于6位', trigger: 'blur'}],
          identity: [{required: false, message: '请确认您的身份', trigger: 'blur'}],
          text: [{required: false, message: '请输入正确的验证码', trigger: 'blur'}],
        },
        msg: ''
      }
    },
    props: ['dialogTableVisible'],
    emits: ['update:dialogTableVisible'],
    methods: {
      close(){
        this.dialog = false
      },

      Info(){
        if(this.checked === true) {
          this.checked=true
          this.dialog = true
        }
      },

      //登录，同时获取用户过去的提交记录
      login() {
        console.log("读取配置文件,url:",url)
        console.log("读取配置文件,upload_url:",upload_url)
        this.form.password = this.password;
        let formData = new FormData();
				formData.append("UserName",this.form.Email);
        formData.append("PassWord",this.form.password);
        axios.post('/login', formData).then(res => {
          if(res.data.message=="登录成功！") {
            this.store.commit('log/UPDATE_UserInfo', this.form.Email)
            this.store.commit('log/UPDATE_LOG', true)
          }
          console.log(res.data.message)
          ElMessageBox.alert(res.data.message, '提示', {
            confirmButtonText: 'OK',
            callback: (action) => {
              ElMessage({
                type: 'info',
                message: res.data.message,
              })
            },
          })
        })
      },
      // 提交给后端用户注册信息
      register() {
        this.form.password = this.password;
        let formData = new FormData();
				formData.append("UserName",this.form.Email);
        formData.append("PassWord",this.form.password);
        formData.append("Identity",this.form.identity);
        formData.append("VerificationCode",this.form.text);
        axios.post('/register', formData).then(res => {
          if(res.data.code == 200) {
            this.store.commit('log/UPDATE_UserInfo', this.form.Email)
            this.store.commit('log/UPDATE_LOG', true)
          }
          console.log(res.data.message)
          ElMessageBox.alert(res.data.message, '提示', {
            confirmButtonText: 'OK',
            callback: (action) => {
              ElMessage({
                type: 'info',
                message: res.data.message,
              })
            },
          })
        })
      },
      ChangePassWord() {
        if(this.pswd_form.newPassWord1 != this.pswd_form.newPassWord2) {
          ElMessage("两次输入的新密码不一致！")
        }
        else{
          let formData = new FormData();
          console.log(this.pswd_form.Email)
          console.log(this.pswd_form.newPassWord1)
          formData.append("UserName", this.pswd_form.Email);
          formData.append("newPassWord", this.pswd_form.newPassWord1);
          formData.append("VerificationCode", this.pswd_form.text);
          axios.post('/forget_password', formData).then(res => {
            console.log(res.data)
            if(res.data.code != 707){
              if(res.data) {
                ElMessageBox.alert(res.data.message, '提示', {
                  confirmButtonText: 'OK',
                  callback: (action) => {
                    ElMessage({
                      type: 'info',
                      message: res.data.message,
                    })
                  },
                })
                if(res.data.code === 200) {
                  this.dlg_forget = false
                }
              }
              else {
                ElMessage("未知错误")
              }
            }
            else{
              this.store.commit('log/UPDATE_LOG', false)
              this.store.commit('log/UPDATE_EXIT', false)
            }
            if (res.data.message) {
              ElMessage(res.data.message)
            }
          })
        }
      },
      exit() {
        this.$emit('update:dialogTableVisible', !this.dialogTableVisible)
      },
      change() {
        this.isRegister = !this.isRegister;
      },
      sendEmail() {
        let formData = new FormData();
				formData.append("UserName",this.form.Email);
        formData.append("Flag", 1);
        axios.post('/verify', formData).then(res => {
          console.log(res.data.message)
          ElMessage(res.data.message)
        })
      },
      sendEmail2() {
        let formData = new FormData();
				formData.append("UserName",this.pswd_form.Email);
        formData.append("Flag", 0);
        axios.post('/verify', formData).then(res => {
          console.log(res.data.message)
          ElMessage(res.data.message)
        })
      },
    }
  }
  </script>
  
  
  <style>
  * {
    margin: 0px;
    padding: 0px;
    box-sizing: border-box;
  }
  
  body {
    background-repeat: no-repeat;
    background-size: 100%;
    background-position: 0px -50px;
  }
  
  .rg_layout1 {
    width: 100%;
    height: 300px;
    border: 1px solid #EEEEEE;
    background-color: white;
    opacity: 0.8;
    /*让div水平居中*/
    margin: auto;
    margin-top: 0px;
  }

  .rg_layout2 {
    width: 100%;
    height: 400px;
    border: 1px solid #EEEEEE;
    background-color: white;
    opacity: 0.8;
    /*让div水平居中*/
    margin: auto;
    margin-top: 0px;
  }
  
  .rg_left {
    float: left;
    margin: 15px;
    width: 20%;
  }
  
  .rg_left > p:first-child {
    color: #FFD026;
    font-size: 20px;
  }
  
  .rg_left > p:last-child {
    color: #A6A6A6;
  }
  
  .rg_center {
    /*border: 1px solid red;*/
    float: left;
    width: 450px;
    /*margin: 15px;*/
  }
  
  .rg_right {
    float: right;
    margin: 15px;
  }
  
  .rg_right > p:first-child {
    font-size: 15px;
  }
  
  .rg_right p a {
    color: pink;
  }
  
  .word-v-middle{
  margin-bottom: 0;
  font-size: 18px;
  min-height: 31px;
  display: flex;
  align-items: center;
  justify-content: center;
  height: 31px;
  margin-top: 5px;
  color: #87878a;
  white-space: normal;
  }
  </style>