import { createApp } from 'vue'
import Vue from 'vue'
import App from './App.vue'
import router from './router/index.js'
// import ElementPlus from 'element-plus'
import * as ElementPlusIconsVue from '@element-plus/icons-vue'
// import 'element-plus/dist/index.css'
import Axios from 'axios'
import '../style/style.css'
import '../static/font/css/siyuan.css'
import store from './store'


// 按需引入element各组件样式
import "element-plus/theme-chalk/el-loading.css";
import "element-plus/theme-chalk/el-message.css";
import "element-plus/theme-chalk/el-notification.css";
import "element-plus/theme-chalk/el-message-box.css";

const app = createApp(App)

let xmlhttp = new XMLHttpRequest();
xmlhttp.open('get', './config.json', false);
xmlhttp.send(null);
let BASE_URL = JSON.parse(xmlhttp.responseText).url
console.log("main.ts:", BASE_URL)

// 挂载axios
// Axios.defaults.baseURL='http://192.168.134.30:5000'
Axios.defaults.baseURL=BASE_URL
// Axios.defaults.baseURL='http://localhost:5000'
// Axios.defaults.baseURL='https://www.lacorrect.cn:8081'
// Axios.defaults.baseURL='http://47.122.17.1:8082'
// Axios.defaults.baseURL='http://117.50.176.44:8082'


app.config.globalProperties.axios = Axios
Axios.defaults.withCredentials = true;
Axios.defaults.timeout = 10 * 60 * 1000

//全局注册icon图标
for (const [key, component] of Object.entries(ElementPlusIconsVue)) {
    app.component(key, component)
  }

app.use(store)

app.use(router)

app.mount('#app')
