import { createRouter, createWebHashHistory } from 'vue-router'

import Submit from '../components/main/submit.vue'
import About from '../components/main/about.vue'
import Detail from '../components/main/result_detail.vue'
import Info from '../components/main/info.vue'
import Privacy from '../components/login/menu_privacy.vue'
import Test from '../components/main/test.vue'

const routes = [
    { path: '/', redirect: '/submit' },
    { path: '/submit', component: Submit },
    { path: '/about', component: About },
    { path: '/detail', component: Detail },
    { path: '/info', component: Info },
    { path: '/privacy', component: Privacy },
    { path: '/test', component: Test },
]

const router = createRouter({
    history: createWebHashHistory(),
    routes,
})

export default router